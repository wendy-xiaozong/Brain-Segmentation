# Some code is borrowed from https://github.com/Project-MONAI/MONAI/blob/master/monai/losses/dice.py

import torch
from pytorch_lightning.metrics.functional import stat_scores_multiple_classes, iou
from pytorch_lightning.utilities import rank_zero_warn, FLOAT16_EPSILON
import scipy.spatial
from typing import Union
from .enums import LossReduction

CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4


def get_score(pred,
              target,
              include_background: bool = True,
              reduction: Union[LossReduction, str] = LossReduction.MEAN) -> torch.tensor:
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

    Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
    """
    num_classes = pred.shape[1]
    # assert num_classes == 139

    tps, fps, tns, fns, sups = stat_scores_multiple_classes(pred, target)
    if not include_background:
        tps = tps[1:]
        fps = fps[1:]
        fns = fns[1:]
        tns = tns[1:]

    dice_denom = (2*tps + fps + fns)
    dice_denom[dice_denom == 0] = torch.tensor(FLOAT16_EPSILON).type_as(dice_denom)
    dice = 2*tps / dice_denom

    iou_denom = fps + fns + tps
    iou_denom[iou_denom == 0] = torch.tensor(FLOAT16_EPSILON).type_as(iou_denom)
    iou = tps / iou_denom

    sensitivity_denom = tps + fns
    sensitivity_denom[sensitivity_denom == 0] = torch.tensor(FLOAT16_EPSILON).type_as(sensitivity_denom)
    sensitivity = tps / sensitivity_denom

    specificity_denom = tns + fps
    specificity_denom[specificity_denom == 0] = torch.tensor(FLOAT16_EPSILON).type_as(specificity_denom)
    specificity = tns / specificity_denom

    print(f"reduction: {reduction}")
    reduction = LossReduction(reduction).value

    if reduction == LossReduction.MEAN.value:
        dice = torch.mean(dice)  # the batch and channel average
        iou = torch.mean(iou)
        sensitivity = torch.mean(sensitivity)
        specificity = torch.mean(specificity)
    # elif reduction == LossReduction.SUM.value:
    #     f = torch.sum(f)  # sum over the batch and channel dims
    elif reduction == LossReduction.NONE.value:
        pass  # returns [N, n_classes] losses
    else:
        raise ValueError(f'Unsupported reduction: {reduction}, available options are ["mean", "sum", "none"].')

    return dice, iou, sensitivity, specificity


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


if __name__ == "__main__":
    # pred = torch.tensor([[0.85, 0.05, 0.05, 0.05],
    #                      [0.05, 0.85, 0.05, 0.05],
    #                      [0.05, 0.05, 0.85, 0.05],
    #                      [0.05, 0.05, 0.05, 0.85]])
    # target = torch.tensor([0, 1, 3, 2])
    # print(get_score(pred, target, include_background=False))
    #
    # target = torch.randint(0, 1, (10, 25, 25))
    # pred = target.clone().detach()
    # pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
    #
    # print(get_score(pred, target, include_background=True))
    # print(get_score(pred, target, include_background=False))
    #
    # print(f"iou: {iou(pred, target)}")
    print(LossReduction(LossReduction.NONE).value)
