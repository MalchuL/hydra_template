import torch.nn as nn
import functools

from src.models.components.layers.group_norm import GroupNorm


def get_pool_layer(pool_type='max'):
    if pool_type == 'max':
        pool_layer = nn.MaxPool2d
    elif pool_type == 'avg':
        pool_layer = nn.AvgPool2d
    else:
        raise NotImplementedError('pool layer [%s] is not found' % pool_type)
    return pool_layer
