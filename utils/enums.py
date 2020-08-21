from enum import Enum


class LossReduction(Enum):
    """
    See also:
        - :py:class:`monai.losses.dice.DiceLoss`
    """

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"
