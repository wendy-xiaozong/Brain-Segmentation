from enum import Enum


# Code is borrowed from: https://github.com/Project-MONAI/MONAI/blob/94a65de0f709d020cfca72140e920468f83f5bd8/monai
# /utils/enums.py#L122
class LossReduction(Enum):
    """
    See also: py:function:`loss.get_score`
    """
    NONE = "none"
    MEAN = "mean"
    SUM = "sum"

