import os
import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.networks.nets import UNETR
from monai.utils import set_determinism
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsChannelLastd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
)
from utils.utils import ConvertToMultiChannelBasedOnBratsClassesd
import glob
import argparse
import time
import nibabel as nib
import random
import numpy as np


torch.manual_seed(10)
random.seed(10)
np.random.seed(10)



parser = argparse.ArgumentParser(description="Transformer segmentation pipeline")
parser.add_argument("--epochs", default=5, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--dataset", default=2020, type=int, help="Dataset to use")
parser.add_argument("--val_frac", default=0.25, type=float, help="fraction of data to use as validation")
parser.add_argument("--num_heads", default=12, type=int, help="Number of heads to use")
parser.add_argument("--embed_dim", default=768, type=int, help="Embedding dimension")


args = parser.parse_args()

root_dir = "./"
set_determinism(seed=0)
device = torch.device("cuda:0")

ds = args.dataset
frac = args.val_frac
max_epochs = args.epochs
batch_size = args.batch_size
num_heads = args.num_heads
embed_dim = args.embed_dim

roi_size = [128, 128, 64]
pixdim = (1.5, 1.5, 2.0)

best_metric = -1
best_metric_epoch = -1

epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

if ds == 2020:
    data_dir = "../Dataset_BRATS_2020/Training/"
elif ds == 2021:
    data_dir = "../Dataset_BRATS_2021/"

t1_list = sorted(glob.glob(data_dir + "*/*t1.nii.gz"))[:100]
t2_list = sorted(glob.glob(data_dir + "*/*t2.nii.gz"))[:100]
t1ce_list = sorted(glob.glob(data_dir + "*/*t1ce.nii.gz"))[:100]
flair_list = sorted(glob.glob(data_dir + "*/*flair.nii.gz"))[:100]
seg_list = sorted(glob.glob(data_dir + "*/*seg.nii.gz"))[:100]

n_data = len(t1_list)

data_dicts = [
    {"images": [t1, t2, t1ce, f], "label": label_name}
    for t1, t2, t1ce, f, label_name in zip(
        t1_list, t2_list, t1ce_list, flair_list, seg_list
    )
]
print("All data: ", len(data_dicts))

random.shuffle(data_dicts)

# for p in data_dicts[0]["images"]:
#     x = nib.load(p).get_fdata(dtype="float32", caching="unchanged")
#     print(x.shape)

# x = nib.load(data_dicts[0]["label"]).get_fdata(dtype="float32", caching="unchanged")
# print(x.shape) 


val_files, train_files = (
    data_dicts[: int(n_data * frac)],
    data_dicts[int(n_data * frac) :],
)

print("Train data: ", len(train_files))
print("Val data: ", len(val_files))

train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["images", "label"]),
        AsChannelFirstd(keys="images", channel_dim=0),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["images", "label"],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["images", "label"], axcodes="RAS"),
        RandSpatialCropd(keys=["images", "label"], roi_size=roi_size, random_size=False),
        RandFlipd(keys=["images", "label"], prob=0.5, spatial_axis=0),
        NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="images", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="images", offsets=0.1, prob=0.5),
        ToTensord(keys=["images", "label"]),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["images", "label"]),
        AsChannelFirstd(keys="images", channel_dim=0),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(
            keys=["images", "label"],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["images", "label"], axcodes="RAS"),
        CenterSpatialCropd(keys=["images", "label"], roi_size=roi_size),
        NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
        ToTensord(keys=["images", "label"]),
    ]
)

train_ds = Dataset(data=train_files, transform=train_transform)
val_ds = Dataset(data=val_files, transform=val_transform)

print("Train Dataset: ", len(train_ds))
print("Val Dataset: ", len(val_ds))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

print("Train Loader: ", len(train_loader))
print("Val Loader: ", len(val_loader))

# model definition    
model = UNETR(
    in_channels=4,
    out_channels=3,
    img_size=tuple(roi_size),
    feature_size=16,
    hidden_size=embed_dim,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)


loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

torch.cuda.empty_cache()

for epoch in range(max_epochs):
    start = time.time()
    print(f"Epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["images"].to(device),
            batch_data["label"].to(device),
        )
        # print(inputs.size(), labels.size())
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"\tAverage loss: {epoch_loss:.4f}")

    # evaluation
    model.eval()
    with torch.no_grad():
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
        post_trans = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
        )
        metric_sum = metric_sum_tc = metric_sum_wt = metric_sum_et = 0.0
        metric_count = metric_count_tc = metric_count_wt = metric_count_et = 0
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["images"].to(device),
                val_data["label"].to(device),
            )
            val_outputs = model(val_inputs)
            val_outputs = post_trans(val_outputs)

            # compute overall mean dice
            value, not_nans = dice_metric(y_pred=val_outputs, y=val_labels).aggregate()
            not_nans = not_nans.mean().item()
            metric_count += not_nans
            metric_sum += value.mean().item() * not_nans
            # compute mean dice for TC
            value_tc, not_nans = dice_metric(
                y_pred=val_outputs[:, 0:1], y=val_labels[:, 0:1]
            ).aggregate()
            not_nans = not_nans.item()
            metric_count_tc += not_nans
            metric_sum_tc += value_tc.item() * not_nans
            # compute mean dice for WT
            value_wt, not_nans = dice_metric(
                y_pred=val_outputs[:, 1:2], y=val_labels[:, 1:2]
            ).aggregate()
            not_nans = not_nans.item()
            metric_count_wt += not_nans
            metric_sum_wt += value_wt.item() * not_nans
            # compute mean dice for ET
            value_et, not_nans = dice_metric(
                y_pred=val_outputs[:, 2:3], y=val_labels[:, 2:3]
            ).aggregate()
            not_nans = not_nans.item()
            metric_count_et += not_nans
            metric_sum_et += value_et.item() * not_nans

        metric = metric_sum / metric_count
        metric_values.append(metric)
        metric_tc = metric_sum_tc / metric_count_tc
        metric_values_tc.append(metric_tc)
        metric_wt = metric_sum_wt / metric_count_wt
        metric_values_wt.append(metric_wt)
        metric_et = metric_sum_et / metric_count_et
        metric_values_et.append(metric_et)
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(
                model.state_dict(),
                os.path.join(root_dir, "best_metric_model.pth"),
            )
            print("\tsaved new best metric model")
        print(
            f"\tMean dice: {metric:.4f}\n"
            f"\ttc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}\n"
            f"\tbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
        )

save_name = "./RESULTS/last.pth"
torch.save(model.state_dict(), save_name)


print(
    f"train completed, best_metric: {best_metric:.4f}" f" at epoch: {best_metric_epoch}"
)
