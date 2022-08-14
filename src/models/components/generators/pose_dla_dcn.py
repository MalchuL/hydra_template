from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from src.models.components.utils.norm_layers import get_norm_layer
from src.models.components.utils.pool_layers import get_pool_layer

logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))

NEG_SLOPE = 0.1
BIAS = False
CONCAT_CHANNELS = True
PADDING_MODE = 'reflect'
POOL_LAYER = nn.MaxPool2d

Conv2d = nn.Conv2d
ConvTransposed2d = nn.ConvTranspose2d


MAIN_ACTIVATION = lambda: nn.ReLU(inplace=True)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=BIAS)
        self.bn1 = norm_layer(bottle_planes)
        self.conv2 = Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                                                                           stride=stride, padding=dilation, padding_mode=PADDING_MODE,
                                                                           bias=BIAS, dilation=dilation)
        self.bn2 = norm_layer(bottle_planes)
        self.conv3 = Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=BIAS)
        self.bn3 = norm_layer(planes)
        self.relu = MAIN_ACTIVATION()
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation, padding_mode=PADDING_MODE,
                               bias=BIAS, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = MAIN_ACTIVATION()
        self.conv2 = Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation, padding_mode=PADDING_MODE,
                               bias=BIAS, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out



class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual, norm_layer=nn.BatchNorm2d):
        super(Root, self).__init__()
        self.conv = Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=BIAS, padding=(kernel_size - 1) // 2, padding_mode=PADDING_MODE, )
        self.bn = norm_layer(out_channels)
        self.relu = MAIN_ACTIVATION()
        self.residual = residual

    def forward(self, *x):
        children = x
        if isinstance(x, (tuple,list)):
            x = list(x)

        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False, norm_layer=nn.BatchNorm2d, pool_layer=POOL_LAYER):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation, norm_layer=norm_layer)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation, norm_layer=norm_layer)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual, norm_layer=norm_layer, pool_layer=pool_layer)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual, norm_layer=norm_layer, pool_layer=pool_layer)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual, norm_layer=norm_layer)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = pool_layer(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=BIAS),
                norm_layer(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x  # We downsample only input, not output
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, input_channels=3, num_classes=1000,
                 block=BasicBlock, residual_root=False, norm_layer=nn.BatchNorm2d, pool_layer=POOL_LAYER):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            Conv2d(input_channels, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=BIAS, padding_mode=PADDING_MODE),
            norm_layer(channels[0]),
            MAIN_ACTIVATION())
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0], norm_layer=norm_layer)
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2, norm_layer=norm_layer)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root, norm_layer=norm_layer, pool_layer=pool_layer)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root, norm_layer=norm_layer, pool_layer=pool_layer)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root, norm_layer=norm_layer, pool_layer=pool_layer)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root, norm_layer=norm_layer, pool_layer=pool_layer)

        # for m in self.modules():
        #     if isinstance(m, Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()


    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        modules = []
        for i in range(convs):
            modules.extend([
                Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=BIAS, dilation=dilation, padding_mode=PADDING_MODE),
                norm_layer(planes),
                MAIN_ACTIVATION()])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla_34_old(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 128, 256],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

def dla_34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 1, 1, 1],
                [32, 64, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

def dla_40(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


def dla_60(pretrained=None, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [32, 64, 128, 128, 512, 1024],
                block=Bottleneck, **kwargs)
    #if pretrained is not None:
    #    model.load_pretrained_model(pretrained, 'dla60')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho, norm_layer=nn.BatchNorm2d):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            norm_layer(cho),
            MAIN_ACTIVATION()
        )
        self.conv = Conv2d(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, groups=1, bias=BIAS, padding_mode=PADDING_MODE)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f, norm_layer=nn.BatchNorm2d):
        super(IDAUp, self).__init__()
        self.channels = channels
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o, norm_layer=norm_layer)
            if CONCAT_CHANNELS:
                node = DeformConv(2 * o, o, norm_layer=norm_layer)
            else:
                node = DeformConv(o, o, norm_layer=norm_layer)
     
            up = ConvTransposed2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=1, bias=False)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)


    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            if CONCAT_CHANNELS:
                layers[i] = node(torch.cat([layers[i], layers[i - 1]], dim=1))
            else:
                layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None, norm_layer=nn.BatchNorm2d):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j], norm_layer=norm_layer))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class DLASeg(nn.Module):
    def __init__(self, base_name, pretrained,
                 head_convs, last_layer, input_channels=3, out_channel=0, norm_layer=nn.BatchNorm2d, pool_layer=nn.MaxPool2d, apply_tanh=False):
        super(DLASeg, self).__init__()

        self.first_level = 0
        self.last_level = last_layer
        self.base = globals()[base_name](pretrained=pretrained, norm_layer=norm_layer, input_channels=input_channels, pool_layer=pool_layer)
        channels = self.base.channels

        scales = [2 ** i for i in range(self.last_level, len(channels))]
        self.dla_up = DLAUp(self.first_level, channels[self.last_level:], scales, norm_layer=norm_layer)

        self.ida_up = IDAUp(channels[self.last_level], channels[self.first_level:self.last_level+1],
                            [2 ** i for i in range(self.last_level - self.first_level+1)], norm_layer=norm_layer)

        self.up = nn.Sequential(Conv2d(channels[self.first_level], channels[self.last_level], 3, bias=BIAS, padding=1, padding_mode=PADDING_MODE), norm_layer(channels[self.last_level]), MAIN_ACTIVATION())

        head = []

        in_out_channels = list(zip([channels[self.last_level]] + head_convs[:],
                                             head_convs[:] + [out_channel]))


        num_no_bn = 1
        for i, (in_channels, out_channels) in enumerate(in_out_channels[:-1]):
            head.extend(
                [Conv2d(in_channels, out_channels, 3, padding=1, bias=BIAS if i < len(in_out_channels[:-1]) - num_no_bn else True, padding_mode=PADDING_MODE),
                 norm_layer(out_channels) if i < len(in_out_channels[:-1]) - num_no_bn else nn.Identity(),
                 nn.CELU(alpha=0.5, inplace=True)])

        head.extend([Conv2d(in_out_channels[-1][0], in_out_channels[-1][1], 3, bias=True, padding=1, padding_mode=PADDING_MODE)])

        self.head = nn.Sequential(*head)
        if apply_tanh:
            self.act = nn.Tanh()
        else:
            self.act = lambda x: x

    def forward(self, x):
        base_x = self.base(x)
        x = self.dla_up(base_x[self.last_level:])
        y = []
        for i in range(self.last_level - self.first_level + 1):
            if i == 0:
                base_x[0] = self.up(base_x[0])
            if i < self.last_level:
                y.append(base_x[i].clone())
            else:
                y.append(x[i-self.last_level].clone())
        self.ida_up(y, 0, len(y))


        x = self.head(y[-1])
        x = self.act(x)
        return x


    
DLA_HEAD_CHANNELS = {'dla_34': [128, 64], 'dla_17': [256, 32], 'dla_60': [256, 64]}


class DLAGenerator(nn.Module):
    def __init__(self, num_layers, output_channels, norm_layer, pool_layer='max', apply_tanh=False, input_channels=3):
        super().__init__()
        dla_name = 'dla_{}'.format(num_layers)
        norm_layer = get_norm_layer(norm_layer)
        pool_layer = get_pool_layer(pool_layer)
        self.model = DLASeg(dla_name, input_channels=input_channels,
                            pretrained=False, last_layer=1,
                            head_convs=DLA_HEAD_CHANNELS[dla_name], out_channel=output_channels, norm_layer=norm_layer, apply_tanh=apply_tanh, pool_layer=pool_layer)

    def forward(self, x):
        return self.model(x)
