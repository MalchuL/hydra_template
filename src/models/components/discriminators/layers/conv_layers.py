import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm as SN


def conv3x3(in_planes, out_planes, stride=1, bias=False, sn=False, use_padding=False):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 if use_padding else 0, bias=bias, padding_mode='reflect')
    if sn:
        conv = SN(conv)
    return conv


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False, sn=False, use_padding=False):
    "1x1 convolution with padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups,
                     padding=1 if use_padding else 0, bias=bias)
    if sn:
        conv = SN(conv)
    return conv


def conv4x4(in_planes, out_planes, stride=1, bias=False, sn=False, use_padding=False):
    "3x3 convolution without padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=stride,
                     padding=2 if use_padding else 0, bias=bias, padding_mode='reflect')
    if sn:
        conv = SN(conv)
    return conv
