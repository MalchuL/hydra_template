import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet34, resnet18
import gc


# VGG_BN should be vgg19_bn-c79401a0.pth


class GANResNetFeatures(nn.Module):

    def __init__(self, network='resnet34', layers=None, real_mean=None, real_std=None, gan_mean=[0.5, 0.5, 0.5], gan_std=[0.225, 0.225, 0.225],
                 fix_pad=False, pooling='max'):
        super().__init__()
        self.pooling = pooling

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
        perception = network_model(pretrained=True).eval()
        self.pre_perception = nn.Sequential(perception.conv1,
                                            perception.bn1,
                                            perception.relu,
                                            perception.maxpool)
        resnet_layers = [*perception.layer1,  # All layers are nn.Sequential
                         *perception.layer2,
                         *perception.layer3,
                         *perception.layer4]
        if layers is None:
            layers = [len(resnet_layers)]
        self.layers = sorted(set(layers))

        self.perception = nn.ModuleList(resnet_layers[:self.layers[-1]])
        if fix_pad:
            self.fix_padding(self.pre_perception)
            self.fix_padding(self.perception)

        for param in self.perception.parameters():
            param.requires_grad = False

        self.perception.requires_grad_(False)

    def fix_padding(self, model: nn.Module, padding='reflect'):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.padding_mode = padding

    def forward(self, x, is_pred):
        self.pre_perception.eval()
        self.perception.eval()
        feats = {'x': x}
        if is_pred:
            x = (x - self.pred_mean) / self.pred_std
        else:
            x = (x - self.real_mean) / self.real_std
        x = self.pre_perception(x)
        for i, module in enumerate(self.perception):
            x = module(x)
            if i in self.layers:
                feats[i] = x
        feats['output'] = x
        return feats

    def extra_repr(self) -> str:
        return f'Real norm: [mean: {self.real_mean.view(3)}, std: mean: {self.real_std.view(3)}], Pred norm: [mean: {self.pred_mean.view(3)}, std: mean: {self.pred_std.view(3)}]' +'\n' +\
            f'layers: {self.layers}'
