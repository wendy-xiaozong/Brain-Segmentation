# Some code is borrowed from https://github.com/Project-MONAI/MONAI/blob/master/monai/losses/dice.py

import torch
from pytorch_lightning.metrics.functional import stat_scores
import scipy.spatial
from typing import Union
from .enums import LossReduction

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4


def get_score(pred,
              target,
              include_background: bool = True,
              reduction: Union[LossReduction, str] = LossReduction.MEAN,
              epsilon=1e-9) -> torch.tensor:
    """
    Args:
        pred: predict tensor
        target: target tensor
        include_background: whether to compute the background class
        reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        epsilon: epsilon
    """
    num_classes = pred.shape[1]

    if not include_background:
        pred = pred[:, 1:]
        target = target[:, 1:]

    assert (
            target.shape == pred.shape
    ), f"ground truth has differing shape ({target.shape}) from input ({pred.shape})"

    nan_score = 0.0
    # do not compute dice for the background
    dice_score = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)
    iou_score = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)
    sensitivity_score = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)
    specificity_score = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)
    for i in range(0, num_classes):
        if not (target == i).any():
            # no foreground class
            dice_score[i] += nan_score
            iou_score[i] += nan_score
            sensitivity_score[i] += nan_score
            specificity_score[i] += nan_score
            continue

        tp, fp, tn, fn, sup = stat_scores(pred=pred, target=target, class_index=i)
        denom = (2 * tp + fp + fn).to(torch.float)

        dice_score_cls = (2 * tp + epsilon).to(torch.float) / denom if torch.is_nonzero(denom) else nan_score
        dice_score[i] += dice_score_cls
        iou_score_cls = (tp + epsilon).to(torch.float) / (tp + fp + fn).to(torch.float) if torch.is_nonzero(denom) \
            else nan_score
        iou_score[i] += iou_score_cls
        sensitivity_score_cls = (tp + epsilon).to(torch.float) / (tp + fn).to(torch.float) if torch.is_nonzero(denom) \
            else nan_score
        sensitivity_score[i] += sensitivity_score_cls
        specificity_score_cls = (tn + epsilon).to(torch.float) / (tn + fp).to(torch.float) if torch.is_nonzero(denom) \
            else nan_score
        specificity_score[i] += specificity_score_cls

    if reduction == LossReduction.MEAN.value:
        dice_score = torch.mean(dice_score)  # the batch and channel average
        iou_score = torch.mean(iou_score)
        sensitivity_score = torch.mean(sensitivity_score)
        specificity_score = torch.mean(specificity_score)
    # elif reduction == LossReduction.SUM.value:
    #     f = torch.sum(f)  # sum over the batch and channel dims
    elif reduction == LossReduction.NONE.value:
        pass  # returns [N, n_classes] losses
    else:
        raise ValueError(f'Unsupported reduction: {reduction}, available options are ["mean", "sum", "none"].')

    return dice_score, iou_score, sensitivity_score, specificity_score


def dice_loss(prob, target):
    """
    code is from https://github.com/CBICA/Deep-BET/blob/master/Deep_BET/utils/losses.py#L11
    :param input:
    :param target:
    :return:
    """
    smooth = 1e-7
    iflat = prob.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))