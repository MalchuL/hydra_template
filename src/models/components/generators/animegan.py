from functools import partial

import torch.nn as nn
import torch

from src.models.components.layers.group_norm import GroupNorm, GroupNorm8
from src.models.components.utils.norm_layers import get_norm_layer

NEG_SLOPE = 0.2
MAIN_ACTIVATION = lambda: nn.LeakyReLU(negative_slope=NEG_SLOPE, inplace=True)


class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            bias=False,
            dilation=1,
            groups=1,
            with_bn=True,
            norm_layer=nn.BatchNorm2d,
            last_act=True,
            padding_mode='reflect'
    ):
        super(conv2DBatchNormRelu, self).__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv_mod = nn.Conv2d(int(in_channels),
                                  int(out_channels),
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  stride=stride,
                                  bias=bias,
                                  groups=groups,
                                  dilation=dilation, padding_mode=padding_mode)

        self.with_bn = with_bn
        if with_bn:
            if norm_layer in (nn.GroupNorm, GroupNorm, GroupNorm8):
                norm_layer = partial(nn.InstanceNorm2d, affine=True)
            self.norm = norm_layer(int(out_channels))
            self.norm_channels = int(out_channels)
        else:
            self.norm = lambda x: x
            self.norm_channels = 0
        self.act = MAIN_ACTIVATION() if last_act else lambda x: x

    def get_norm_channels(self):
        return self.norm_channels

    def forward(self, inputs):
        outputs = self.act(self.norm(self.conv_mod(inputs)))
        return outputs


class InvertibleBlock(nn.Module):
    def __init__(self, in_channel, expansion_ratio, output_dim, bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        bottleneck_dim = round(expansion_ratio * in_channel)
        self.net = nn.Sequential(conv2DBatchNormRelu(in_channel, bottleneck_dim, kernel_size=1, bias=bias, norm_layer=norm_layer),
                                 conv2DBatchNormRelu(bottleneck_dim, bottleneck_dim, bias=bias, groups=bottleneck_dim, norm_layer=norm_layer),
                                 conv2DBatchNormRelu(bottleneck_dim, output_dim, kernel_size=1, bias=bias, last_act=False, norm_layer=norm_layer))

    def forward(self, x):
        out = self.net(x)
        if x.shape[1] == out.shape[1]:
            out = out + x
        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_bilinear=True, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.layer = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear') if use_bilinear else nn.ConvTranspose2d(in_channels, in_channels, 2 * 2, stride=2, padding=2 // 2, output_padding=0, bias=False),
            conv2DBatchNormRelu(in_channels, out_channels, kernel_size=kernel_size, norm_layer=norm_layer))

    def forward(self, x):
        return self.layer(x)


# From https://github.com/TachibanaYoshino/AnimeGANv2/blob/d43b99a963f20af4d08e0d2c7a54ee7c06772867/net/generator.py#L31
class AnimeGAN(nn.Module):
    def __init__(self, in_channels=3, apply_tanh=True, norm_layer='batch', multiplier=1):
        super().__init__()
        norm_layer = get_norm_layer(norm_layer)
        def r(num_channels):
            return round(num_channels * multiplier)
        # A
        self.A_block = nn.Sequential(conv2DBatchNormRelu(in_channels, r(32), 7, norm_layer=norm_layer),
                                     conv2DBatchNormRelu(r(32), r(64), stride=2, norm_layer=norm_layer),
                                     conv2DBatchNormRelu(r(64), r(64), norm_layer=norm_layer))
        # B
        self.B_block = nn.Sequential(conv2DBatchNormRelu(r(64), r(128), stride=2, norm_layer=norm_layer),
                                     conv2DBatchNormRelu(r(128), r(128), norm_layer=norm_layer))
        # C
        self.C_block = nn.Sequential(conv2DBatchNormRelu(r(128), r(128), norm_layer=norm_layer),
                                     InvertibleBlock(r(128), 2, r(256), norm_layer=norm_layer),
                                     InvertibleBlock(r(256), 2, r(256), norm_layer=norm_layer),
                                     InvertibleBlock(r(256), 2, r(256), norm_layer=norm_layer),
                                     InvertibleBlock(r(256), 2, r(256), norm_layer=norm_layer),
                                     conv2DBatchNormRelu(r(256), r(128), norm_layer=norm_layer))

        # D
        self.D_block = nn.Sequential(Upsample(r(128), r(128), norm_layer=norm_layer),
                                     conv2DBatchNormRelu(r(128), r(128), norm_layer=norm_layer))

        # E
        self.E_block = nn.Sequential(Upsample(r(128), r(64), norm_layer=norm_layer),
                                     conv2DBatchNormRelu(r(64), r(64), norm_layer=norm_layer),
                                     conv2DBatchNormRelu(r(64), r(32), kernel_size=7, norm_layer=norm_layer))
        # out_layer
        self.out = nn.Sequential(nn.Conv2d(r(32), in_channels, kernel_size=1, bias=False),
                                 nn.Tanh() if apply_tanh else nn.Identity())


    def forward(self, x):
        x = self.A_block(x)
        x = self.B_block(x)
        x = self.C_block(x)
        x = self.D_block(x)
        x = self.E_block(x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    x = torch.rand(5,3,256,256)
    model = AnimeGAN(norm_layer='batch')
    print(model)
    print(model(x).shape)