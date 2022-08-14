import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.layers.sobel import SobelFilter
from .layers.conv_layers import conv3x3 as conv3x3, conv1x1 as conv1x1, conv4x4

NEG_SLOPE = 0.2
LAST_BIAS = False

class StdPool(nn.Module):
    def forward(self, x):
        return torch.std(x, dim=(2, 3), keepdim=True)

class MinPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((1,1))
    def forward(self, x):
        return self.pool(-x)


class ChannelsPool(nn.Module):

    def __init__(self, in_channels, hidden_layers, out_layers, eps=1e-5, channels_as_groups=True, pool_layers=('avg', 'max', 'std')):
        super().__init__()
        self.eps = eps
        layers = {'avg': nn.AdaptiveAvgPool2d((1, 1)),
                  'max': nn.AdaptiveMaxPool2d((1, 1)),
                  'std': StdPool(),
                  'min': MinPool()}
        self.stats = nn.ModuleList(
            [layers[pool_layer] for pool_layer in pool_layers])
        groups = len(pool_layers) if channels_as_groups else 1
        in_channels = in_channels * len(pool_layers)
        hidden_layers = hidden_layers * groups
        out_layers = out_layers * groups

        def get_act():
            return nn.CELU(alpha=0.5, inplace=True)  # alpha because at -1 gradient lokks like leaky relu
        def get_norm(channels, affine=True):
            return nn.BatchNorm2d(channels, affine=affine, eps=self.eps)
        self.mlp = nn.Sequential(conv1x1(in_channels, hidden_layers, groups=groups, bias=False),
                                 get_norm(hidden_layers),
                                 get_act(),
                                 conv1x1(hidden_layers, hidden_layers, groups=groups, bias=False),
                                 get_norm(hidden_layers),
                                 get_act(),
                                 conv1x1(hidden_layers, out_layers, groups=groups, bias=LAST_BIAS),
                                 )

    def forward(self, x):
        x = [stat(x) for stat in self.stats]
        x = torch.cat(x, 1)
        x = self.mlp(x)
        return x


class MinibatchSTD(nn.Module):
    def calculate_std(self, x):
        return torch.std(x, dim=(1), keepdim=True)

    @staticmethod
    def get_num_channels():
        return 1

    def forward(self, x):
        return torch.cat([x, self.calculate_std(x)], dim=1)


# Don't change LeakyRelu
class CartoonGANDiscriminator(nn.Module):
    def __init__(self, use_sigmoid,
                 layers=[32, 64, 128, 128, 256],
                 strides=[1, 2, 1, 2, 1],
                 mlp=[1, 3, 5],
                 mean=[0.5, 0.5, 0.5],
                 std=[0.225, 0.225, 0.225],
                 multiplier=1,
                 apply_sobel=True,
                 eps=1e-5,
                 channels_as_groups=True,
                 use_padding=True,
                 apply_instance_norm=False):
        super(CartoonGANDiscriminator, self).__init__()

        self.eps = eps
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))

        self.sobel = SobelFilter(use_padding=use_padding, mean=mean, std=std)  # Apply sobel filter over input
        SOBEL_APPLY_LAYER = 1 if apply_sobel else -1
        self.sobel_apply = -1

        def r(x):
            return round(x * multiplier)


        def get_act(inplace=True):
            return nn.CELU(alpha=0.5, inplace=inplace)  # alpha because at -1 gradient lokks like leaky relu

        def get_norm(channels, affine=True):
            return nn.InstanceNorm2d(channels, affine=affine) if apply_instance_norm else nn.BatchNorm2d(channels, affine=affine, eps=self.eps, track_running_stats=False)

        def get_reverse_norm(channels, affine=True):
            return nn.Identity()

        layers = [r(layer) for layer in layers]
        in_channels = [3] + layers[:-1]
        out_channels = layers
        strides = list(strides)
        modules = []
        self.mlp = nn.ModuleDict()
        for i in range(len(strides)):
            if i == SOBEL_APPLY_LAYER:
                in_channels[i] += self.sobel.get_num_channels()
                self.sobel_apply = len(modules)
            if i in mlp:
                self.mlp[str(len(modules))] = ChannelsPool(out_channels[i], out_channels[i], 1, eps=self.eps, channels_as_groups=channels_as_groups)
            use_norm = i > 0 and strides[i] == 1
            conv_fn = conv3x3
            modules.extend(
                [conv_fn(in_channels[i], out_channels[i], stride=strides[i], bias=not use_norm, sn=False, use_padding=use_padding),
                 get_norm(out_channels[i], affine=True) if use_norm else get_reverse_norm(out_channels[i], affine=True),
                 get_act(inplace=use_norm)])


        # get ProGAN std
        minibatch_std = MinibatchSTD()
        modules.append(minibatch_std)
        modules.append(conv3x3(out_channels[-1] + minibatch_std.get_num_channels(), 1, stride=1, bias=LAST_BIAS, use_padding=use_padding)) # You must have bias as true to good converge
        self.use_sigmoid = use_sigmoid
        self.conv = nn.ModuleList(modules)


    def forward(self, x, apply_sigmoid=True):
        grad = self.sobel(x)
        x = (x - self.mean) / self.std
        stats = []
        for i, module in enumerate(self.conv):
            if i == self.sobel_apply:
                x = torch.cat([x, grad], dim=1)
            x = module(x)
            if str(i) in self.mlp:
                stats.append(self.mlp[str(i)](x))

        if len(stats) > 0:
            stats = torch.cat(stats, dim=1)
        if self.use_sigmoid and apply_sigmoid:
            x = torch.sigmoid(x)
            if len(stats) > 0:
                stats = torch.sigmoid(stats)

        if len(stats) == 0:
            stats = None
        return x, stats


if __name__ == '__main__':
    x = torch.rand(5, 32, 128, 128)
    mod = ChannelsPool(32, 128, 1)
    print(mod(x).shape)
