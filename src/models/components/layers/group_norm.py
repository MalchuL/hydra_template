from math import sqrt, log2

import torch.nn as nn
import functools

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int, affine=True):
        # num_channels = groups * (groups/ DIVIDER)
        groups = 2 ** round((log2(sqrt(num_channels * 2))))
        num_groups = groups
        super().__init__(num_groups, num_channels, affine=affine)


class GroupNorm8(nn.GroupNorm):
    def __init__(self, num_channels: int, affine=True):
        # num_channels = groups * (groups/ DIVIDER)
        groups = max(8, num_channels // 8)
        num_groups = groups
        super().__init__(num_groups, num_channels, affine=affine)