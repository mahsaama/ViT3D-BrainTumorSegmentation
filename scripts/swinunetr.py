import os
import torch
from monai.data import DataLoader, Dataset
from monai.losses.dice import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNETR
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
from utils.utils import (
    ConvertToMultiChannelBasedOnBratsClassesd,
    sec_to_minute,
    LinearWarmupCosineAnnealingLR,
    # SimCLR_Loss,
    # SupervisedContrastiveLoss,
    # mixup_data,
    # augment_rare_classes,
)
import glob
import argparse
import time
import random
import numpy as np
import warnings
import nibabel as nib
import SimpleITK as sitk


warnings.filterwarnings("ignore")

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

parser = argparse.ArgumentParser(description="Transformer segmentation pipeline")
parser.add_argument(
    "--epochs", default=5, type=int, help="max number of training epochs"
)
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--dataset", default="2020", type=str, help="Dataset to use")
parser.add_argument(
    "--val_frac", default=0.25, type=float, help="fraction of data to use as validation"
)
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

roi_size = [128, 128, 64]  # TODO: change 64 to 128
pixdim = (1.5, 1.5, 2.0)

best_metric = -1
best_metric_epoch = -1

epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

if ds == "2020":
    data_dir = "../Dataset_BRATS_2020/Training/"
    t1_list = sorted(glob.glob(data_dir + "*/*t1.nii.gz"))
    t2_list = sorted(glob.glob(data_dir + "*/*t2.nii.gz"))
    t1ce_list = sorted(glob.glob(data_dir + "*/*t1ce.nii.gz"))
    flair_list = sorted(glob.glob(data_dir + "*/*flair.nii.gz"))
    seg_list = sorted(glob.glob(data_dir + "*/*seg.nii.gz"))
    #
    # data_dir = "../Dataset_BRATS_2020/Augmented/"
    # t1_list += sorted(glob.glob(data_dir + "*/*t1.nii.gz"))
    # t2_list += sorted(glob.glob(data_dir + "*/*t2.nii.gz"))
    # t1ce_list += sorted(glob.glob(data_dir + "*/*t1ce.nii.gz"))
    # flair_list += sorted(glob.glob(data_dir + "*/*flair.nii.gz"))
    # seg_list += sorted(glob.glob(data_dir + "*/*seg.nii.gz"))

elif ds == "2021":
    data_dir = "../Dataset_BRATS_2021/"
    t1_list = sorted(glob.glob(data_dir + "*/*t1.nii.gz"))
    t2_list = sorted(glob.glob(data_dir + "*/*t2.nii.gz"))
    t1ce_list = sorted(glob.glob(data_dir + "*/*t1ce.nii.gz"))
    flair_list = sorted(glob.glob(data_dir + "*/*flair.nii.gz"))
    seg_list = sorted(glob.glob(data_dir + "*/*seg.nii.gz"))
elif ds == "2020-2021":  # combiantion of 2020 and 2021, TODO: remove
    data_dir = "../Dataset_BRATS_2020/Training/"
    t1_list = sorted(glob.glob(data_dir + "*/*t1.nii.gz"))
    t2_list = sorted(glob.glob(data_dir + "*/*t2.nii.gz"))
    t1ce_list = sorted(glob.glob(data_dir + "*/*t1ce.nii.gz"))
    flair_list = sorted(glob.glob(data_dir + "*/*flair.nii.gz"))
    seg_list = sorted(glob.glob(data_dir + "*/*seg.nii.gz"))
    data_dir = "../Dataset_BRATS_2021/"
    t1_list += sorted(glob.glob(data_dir + "*/*t1.nii.gz"))
    t2_list += sorted(glob.glob(data_dir + "*/*t2.nii.gz"))
    t1ce_list += sorted(glob.glob(data_dir + "*/*t1ce.nii.gz"))
    flair_list += sorted(glob.glob(data_dir + "*/*flair.nii.gz"))
    seg_list += sorted(glob.glob(data_dir + "*/*seg.nii.gz"))

n_data = len(t1_list)
print(n_data)

data_dicts = [
    {"images": [t1, t2, t1ce, f], "label": label_name}
    for t1, t2, t1ce, f, label_name in zip(
        t1_list, t2_list, t1ce_list, flair_list, seg_list
    )
]

random.shuffle(data_dicts)

num_0 = 0
num_1 = 0
num_2 = 0
num_4 = 0

for p in seg_list:
    image = sitk.ReadImage(p)
    arr = sitk.GetArrayViewFromImage(image)
    values, counts = np.unique(arr, return_counts=True)
    num_0 += counts[0]
    num_1 += counts[1]
    num_2 += counts[2]
    num_4 += counts[3]

print(num_0)
print(num_1)
print(num_2)
print(num_4)

# for p in data_dicts[0]["label"]:
# #     x = nib.load(p).get_fdata(dtype="float32", caching="unchanged")
# #     print(x.shape)
#
#     x = nib.load(p).get_fdata(dtype="float32", caching="unchanged")
#     print(type(x))


val_files, train_files = (
    data_dicts[: int(n_data * frac)],
    data_dicts[int(n_data * frac):],
)

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
        RandSpatialCropd(
            keys=["images", "label"], roi_size=roi_size, random_size=False
        ),
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

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

# model definition
# model = UNETR(
#     in_channels=4,
#     out_channels=3,
#     img_size=tuple(roi_size),
#     feature_size=48,
#     hidden_size=embed_dim,
#     mlp_dim=3072,
#     num_heads=num_heads,
#     pos_embed="perceptron",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
# ).to(device)


# class weights
class_weights = np.array([45.465614, 16.543337, 49.11155], dtype="f")
weights = torch.tensor(
    class_weights, dtype=torch.float32, device=torch.device("cuda:0")
)

model = SwinUNETR(
    img_size=tuple(roi_size),
    in_channels=4,
    out_channels=3,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=False,
).to(device)

weight = torch.load("./model_swinvit.pt")
model.load_from(weights=weight)
print("Using pretrained self-supervied Swin UNETR backbone weights!")

# for name, param in model.named_parameters():
#     if "swinViT" in name and "layers" in name:
#         param.requires_grad = False

loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True, ce_weight=weights)
# loss_function = SupervisedContrastiveLoss()
# loss_function = SimCLR_Loss(batch_size, 0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
scheduler = LinearWarmupCosineAnnealingLR(
    optimizer, warmup_epochs=1, max_epochs=max_epochs
)
torch.cuda.empty_cache()

results_path = os.path.join(".", "RESULTS")
if os.path.exists(results_path) == False:
    os.mkdir(results_path)

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
        # print(step)
        # print(inputs.size())
        # print(labels.size())

        # inputs, labels = augment_rare_classes(inputs, labels)
        # xs_mixup, ys_mixup_a, ys_mixup_b, lam = mixup_data(
        #     x=inputs,
        #     y=labels,
        #     alpha=1)

        # print(torch.unique(labels))
        optimizer.zero_grad()
        try:
            outputs = model(inputs)
        except Exception as e:
            print(step)
            print(e)
            continue
        # print(outputs.size(), labels.size())

        # loss = lam * loss_function(outputs, ys_mixup_a) + (1 - lam) * loss_function(outputs, ys_mixup_b)

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
        dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=True
        )
        post_trans = Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(threshold=0.6),
            ]
        )
        metric_sum = metric_sum_tc = metric_sum_wt = metric_sum_et = 0.0
        metric_count = metric_count_tc = metric_count_wt = metric_count_et = 0
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["images"].to(device),
                val_data["label"].to(device),
            )
            try:
                val_outputs = model(val_inputs)
            except Exception as e:
                print(e)
                continue
            val_outputs = post_trans(val_outputs)
            dice_metric(y_pred=val_outputs, y=val_labels)

            # compute overall mean dice
            value, not_nans = dice_metric.aggregate()
            dice_metric.reset()
            not_nans = not_nans.mean().item()
            metric_count += not_nans
            metric_sum += value.mean().item() * not_nans

            # compute mean dice for TC
            dice_metric(y_pred=val_outputs[:, 1:2], y=val_labels[:, 1:2])
            value_tc, not_nans = dice_metric.aggregate()
            dice_metric.reset()
            not_nans = not_nans.item()
            metric_count_tc += not_nans
            metric_sum_tc += value_tc.item() * not_nans

            # compute mean dice for WT
            dice_metric(y_pred=val_outputs[:, 2:3], y=val_labels[:, 2:3])
            value_wt, not_nans = dice_metric.aggregate()
            dice_metric.reset()
            not_nans = not_nans.item()
            metric_count_wt += not_nans
            metric_sum_wt += value_wt.item() * not_nans

            # compute mean dice for ET
            dice_metric(y_pred=val_outputs[:, 3:4], y=val_labels[:, 3:4])
            value_et, not_nans = dice_metric.aggregate()
            dice_metric.reset()
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
            f"\tTC: {metric_tc:.4f} WT: {metric_wt:.4f} ET: {metric_et:.4f}\n"
            f"\tBest mean dice: {best_metric:.4f} at Epoch: {best_metric_epoch}\n"
            f"\tTime: {sec_to_minute(time.time() - start)}"
        )
    scheduler.step()

save_name = "./RESULTS/last.pth"
torch.save(model.state_dict(), save_name)

print(
    f"Train completed, best_metric: {best_metric:.4f}" f" at epoch: {best_metric_epoch}"
)
