# Some code is from: https://github.com/Project-MONAI/MONAI/blob/master/monai/losses/dice.py
import torch
from torch import tensor
from typing import Union
from .enums import LossReduction
from monai.networks import one_hot
import warnings


def dice_loss(input: tensor,
              target: tensor,
              include_background: bool = True,
              softmax: bool = False,
              to_onehot: bool = True,
              squared_pred: bool = False,
              reduction: Union[LossReduction, str] = LossReduction.MEAN,
              smooth: float = 1e-5):
    """
    loss function, from
    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.

    Args:
        input: predict tensor，the shape should be BNH[WD].
        target: target tensor, the shape should be BNH[WD].
        include_background:
        softmax: if True, apply a softmax function to the prediction.
        to_onehot: whether to convert `target` into the one-hot format. Defaults to False.
        squared_pred: use squared versions of targets and predictions in the denominator or not.
        reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        smooth: a small constant to avoid nan.
    """

    n_pred_ch = input.shape[1]
    if softmax:
       input = torch.softmax(input, 1)

    if to_onehot:
        if n_pred_ch == 1:
            warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
        else:
            # the F.one_hot can not use here, because it would return BNH[WD]C (C is the class
            target = one_hot(target.to(torch.int64), num_classes=n_pred_ch)

    if not include_background:
        if n_pred_ch == 1:
            warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            # if skipping background, removing first channel
            target = target[:, 1:]
            input = input[:, 1:]

    assert (
            target.shape == input.shape
    ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

    # reducing only spatial dimensions (not batch nor channels)
    reduce_axis = list(range(2, len(input.shape)))
    intersection = torch.sum(target * input, dim=reduce_axis)

    if squared_pred:
        target = torch.pow(target, 2)
        input = torch.pow(input, 2)

    ground_o = torch.sum(target, dim=reduce_axis)
    pred_o = torch.sum(input, dim=reduce_axis)

    denominator = ground_o + pred_o

    f = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

    reduction = LossReduction(reduction).value
    if reduction == LossReduction.MEAN.value:
        f = torch.mean(f)  # the batch and channel average
    elif reduction == LossReduction.SUM.value:
        f = torch.sum(f)  # sum over the batch and channel dims
    elif reduction == LossReduction.NONE.value:
        pass  # returns [N, n_classes] losses
    else:
        raise ValueError(f'Unsupported reduction: {reduction}, available options are ["mean", "sum", "none"].')

    return f

