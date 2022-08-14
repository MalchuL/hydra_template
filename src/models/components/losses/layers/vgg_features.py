import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn as vgg19_bn, vgg19
import gc


# VGG_BN should be vgg19_bn-c79401a0.pth


class Scale(nn.Module):
    def __init__(self, module, scale):
        super().__init__()
        self.module = module
        self.register_buffer('scale', torch.tensor(scale))

    def extra_repr(self):
        return f'(scale): {self.scale.item():g}'

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) * self.scale


class VGGFeatures(nn.Module):
    poolings = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d, 'l2': partial(nn.LPPool2d, 2)}
    #pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}
    pooling_scales = {'max': 1., 'average': 1, 'l2': 0.78}
    layers_mapping = {'vgg19': 26, 'vgg19_bn': 37}

    def __init__(self, network='vgg19_bn', fix_pad=False, pooling='max'):
        super().__init__()
        self.pooling = pooling

        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))

        network_model = globals()[network]
        perception = list(network_model(pretrained=True).features)[:self.layers_mapping[network]]
        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(perception):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                perception[i] = Scale(self.poolings[pooling](2), pool_scale)
        self.perception = nn.Sequential(*perception).eval()

        if fix_pad:
            self.fix_padding(self.perception)
        torch.nn.init.constant_(self.perception[-1].bias, 0)
        for param in self.perception.parameters():
            param.requires_grad = False

        self.perception.requires_grad_(False)

    def fix_padding(self, model: nn.Module, padding='reflect'):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.padding_mode = padding
                module.padding = 0

    def forward(self, x):
        self.perception.eval()
        x = (x - self.mean) / self.std
        return self.perception(x)


class GANVGGFeatures(nn.Module):
    poolings = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d, 'l2': partial(nn.LPPool2d, 2)}
    # pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}
    pooling_scales = {'max': 1., 'average': 1, 'l2': 0.78}
    layers_mapping = {'vgg19': 26, 'vgg19_bn': 37}

    def __init__(self, network='vgg19_bn', layers=None, real_mean=None, real_std=None, gan_mean=[0.5, 0.5, 0.5], gan_std=[0.225, 0.225, 0.225], fix_pad=False, pooling='max'):
        super().__init__()
        self.pooling = pooling
        if layers is None:
            layers=[self.layers_mapping[network]-1]
        self.layers = sorted(set(layers))

        if gan_mean or gan_std:
            assert gan_mean and gan_std, (gan_mean, gan_std)
            self.register_buffer('pred_std', torch.tensor(gan_std).view(1, -1, 1, 1))
            self.register_buffer('pred_mean', torch.tensor(gan_mean).view(1, -1, 1, 1))
        else:
            self.register_buffer('pred_std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
            self.register_buffer('pred_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))

        if real_mean or real_std:
            assert real_mean and real_std, (real_mean, real_std)
            self.register_buffer('real_std', torch.tensor(real_std).view(1, -1, 1, 1))
            self.register_buffer('real_mean', torch.tensor(real_mean).view(1, -1, 1, 1))
        else:
            self.register_buffer('real_std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
            self.register_buffer('real_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))

        network_model = globals()[network]
        perception = list(network_model(pretrained=True).features)[:self.layers_mapping[network]]
        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(perception):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                perception[i] = Scale(self.poolings[pooling](2), pool_scale)
        self.perception = nn.Sequential(*perception).eval()

        self.fix_padding(self.perception, fix_pad)

        torch.nn.init.constant_(self.perception[-1].bias, 0)
        for param in self.perception.parameters():
            param.requires_grad = False

        self.perception.requires_grad_(False)

    def fix_padding(self, model: nn.Module, fix_pad=True, padding='reflect'):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                if fix_pad:
                    module.padding_mode = padding
                else:
                    module.padding = (0,) * len(module.padding)

    def forward(self, x, is_pred):
        self.perception.eval()
        feats = {'x': x}
        if is_pred:
            x = (x - self.pred_mean) / self.pred_std
        else:
            x = (x - self.real_mean) / self.real_std
        for i, module in enumerate(self.perception):
            x = module(x)
            if i in self.layers:
                feats[i] = x
        feats['output'] = x
        return feats

    def extra_repr(self) -> str:
        return f'Real norm: [mean: {self.real_mean.view(3)}, std: mean: {self.real_std.view(3)}], Pred norm: [mean: {self.pred_mean.view(3)}, std: mean: {self.pred_std.view(3)}]' +'\n' +\
            f'layers: {self.layers}'
