from monai.transforms import MapTransform
import numpy as np
import math
import warnings
from typing import List
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch import nn as nn
from torch.optim import Optimizer
import torch
from math import log


__all__ = ["LinearLR", "ExponentialLR"]


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))


# class SupervisedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         """
#         Implementation of the loss described in the paper Supervised Contrastive Learning :
#         https://arxiv.org/abs/2004.11362
#
#         :param temperature: int
#         """
#         super(SupervisedContrastiveLoss, self).__init__()
#         self.temperature = temperature
#
#     def forward(self, projections, targets):
#         """
#
#         :param projections: torch.Tensor, shape [batch_size, projection_dim]
#         :param targets: torch.Tensor, shape [batch_size]
#         :return: torch.Tensor, scalar
#         """
#         device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
#         print(projections.shape)
#         print(projections.T.shape)
#
#         print(torch.mm(projections, projections.T))
#         dot_product_tempered = torch.mm(projections, projections.mT) / (
#             self.temperature
#         )
#         # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
#         exp_dot_tempered = (
#             torch.exp(
#                 dot_product_tempered
#                 - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]
#             )
#             + 1e-5
#         )
#
#         mask_similar_class = (
#             targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets
#         ).to(device)
#         mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
#         mask_combined = mask_similar_class * mask_anchor_out
#         cardinality_per_samples = torch.sum(mask_combined, dim=1)
#
#         log_prob = -torch.log(
#             exp_dot_tempered
#             / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True))
#         )
#         supervised_contrastive_loss_per_sample = (
#             torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
#         )
#         supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
#
#         return supervised_contrastive_loss


class BinaryLabel_WT(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            # result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1)
            )
            # label 2 is ET
            # result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class _LRSchedulerMONAI(_LRScheduler):
    """Base class for increasing the learning rate between two boundaries over a number
    of iterations"""

    def __init__(
        self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1
    ) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            end_lr: the final learning rate.
            num_iter: the number of iterations over which the test occurs.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_LRSchedulerMONAI, self).__init__(optimizer, last_epoch)


class LinearLR(_LRSchedulerMONAI):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRSchedulerMONAI):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Based on https://huggingface.co/ implementation.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        t_total: int,
        cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            warmup_steps: number of warmup iterations.
            t_total: total number of training iterations.
            cycles: cosine cycles parameter.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1)
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


def sec_to_minute(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)) % 24)

    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
