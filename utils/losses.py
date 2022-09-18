
import warnings
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            target_1 = target[:, 0]
            input_1 = input[:, 0]
            target_2 = target[:, 1]
            input_2 = input[:, 1]
            target_3 = target[:, 2]
            input_3 = input[:, 2]
        else:
            # if skipping background, removing first channel
            target_1 = target[:, 1]
            input_1 = input[:, 1]
            target_2 = target[:, 2]
            input_2 = input[:, 2]
            target_3 = target[:, 3]
            input_3 = input[:, 3]

        print(input.shape)
        print(target.shape)

        if target_1.shape != input_1.shape:
            raise AssertionError(f"1. ground truth has different shape ({target_1.shape}) from input ({input_1.shape})")
        if target_2.shape != input_2.shape:
            raise AssertionError(f"2. ground truth has different shape ({target_2.shape}) from input ({input_2.shape})")
        if target_3.shape != input_3.shape:
            raise AssertionError(f"3. ground truth has different shape ({target_3.shape}) from input ({input_3.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis1: List[int] = torch.arange(2, len(input_1.shape)).tolist()
        reduce_axis2: List[int] = torch.arange(2, len(input_2.shape)).tolist()
        reduce_axis3: List[int] = torch.arange(2, len(input_3.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis1 = [0] + reduce_axis1
            reduce_axis2 = [0] + reduce_axis2
            reduce_axis3 = [0] + reduce_axis3


        intersection1 = torch.sum(target * input, dim=reduce_axis1)
        intersection2 = torch.sum(target * input, dim=reduce_axis2)
        intersection3 = torch.sum(target * input, dim=reduce_axis3)


        if self.squared_pred:
            target_1 = torch.pow(target_1, 2)
            input_1 = torch.pow(input_1, 2)
            target_2 = torch.pow(target_2, 2)
            input_2 = torch.pow(input_2, 2)
            target_3 = torch.pow(target_3, 2)
            input_3 = torch.pow(input_3, 2)

        ground_o_1 = torch.sum(target, dim=reduce_axis1)
        pred_o_1 = torch.sum(input, dim=reduce_axis1)
        ground_o_2 = torch.sum(target, dim=reduce_axis2)
        pred_o_2 = torch.sum(input, dim=reduce_axis2)
        ground_o_3 = torch.sum(target, dim=reduce_axis3)
        pred_o_3 = torch.sum(input, dim=reduce_axis3)

        denominator1 = ground_o_1 + pred_o_1
        denominator2 = ground_o_2 + pred_o_2
        denominator3 = ground_o_3 + pred_o_3

        if self.jaccard:
            denominator1 = 2.0 * (denominator1 - intersection1)
            denominator2 = 2.0 * (denominator2 - intersection2)
            denominator3 = 2.0 * (denominator3 - intersection3)

        f1: torch.Tensor = 1.0 - (2.0 * intersection1 + self.smooth_nr) / (denominator1 + self.smooth_dr)
        f2: torch.Tensor = 1.0 - (2.0 * intersection2 + self.smooth_nr) / (denominator2 + self.smooth_dr)
        f3: torch.Tensor = 1.0 - (2.0 * intersection3 + self.smooth_nr) / (denominator3 + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f1 = torch.mean(f1)  # the batch and channel average
            f2 = torch.mean(f2)  # the batch and channel average
            f3 = torch.mean(f3)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f1 = torch.sum(f1)  # sum over the batch and channel dims
            f2 = torch.sum(f2)  # sum over the batch and channel dims
            f3 = torch.sum(f3)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f1.shape[0:2]) + [1] * (len(input_1.shape) - 2)
            f1 = f1.view(broadcast_shape)

            broadcast_shape = list(f2.shape[0:2]) + [1] * (len(input_2.shape) - 2)
            f2 = f2.view(broadcast_shape)

            broadcast_shape = list(f3.shape[0:2]) + [1] * (len(input_3.shape) - 2)
            f3 = f3.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        x = f1 + f2 + f3
        return (1+(f1/x))*f1 + (1+(f2/x))*f2 + (1+(f3/x))*f3


class DiceCELoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example: `other_act = torch.tanh`.
                only used by the `DiceLoss`, don't need to specify activation function for `CrossEntropyLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def ce(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        return self.cross_entropy(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss