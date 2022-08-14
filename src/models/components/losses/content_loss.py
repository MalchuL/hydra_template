from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.losses.layers.resnet_features import GANResNetFeatures
from src.models.components.losses.layers.vgg_features import VGGFeatures, GANVGGFeatures


class PretrainContentLoss(nn.Module):
    def __init__(self):
        super(PretrainContentLoss, self).__init__()
        self.base_loss = nn.L1Loss()

        self.perception = VGGFeatures(fix_pad=True)

    def forward(self, pred, target):
        pred = self.perception(pred)
        with torch.no_grad():
            target = self.perception(target)

        return self.base_loss(pred, target)

class ContentLoss(nn.Module):
    vgg_19_to_bn = {0: 0,
                    2: 3,
                    5: 7,
                    7: 10,
                    10: 14,
                    12: 17,
                    14: 20,
                    16: 23,
                    19: 27,
                    21: 30,
                    23: 33,
                    25: 36}
    # Layers from https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa
    # Layers mapped on vgg19
    def __init__(self, model_name='vgg19_bn', layers=(), weight_scaler=2, apply_norm=False, fix_pad=False, real_mean=None, real_std=None, pred_mean=[0.5, 0.5, 0.5], pred_std=[0.225, 0.225, 0.225]):
        super(ContentLoss, self).__init__()

        self.pred_std = pred_std
        self.pred_mean = pred_mean
        self.apply_norm = apply_norm
        self.base_loss = self.get_loss()
        self.perception = self.get_model(model_name=model_name, layers=layers, fix_pad=fix_pad, real_mean=real_mean, real_std=real_std, pred_mean=pred_mean, pred_std=pred_std)

        weights = list(reversed([1 / (weight_scaler ** i) for i in range(len(layers) + 1)]))
        sum_weight = sum(weights)
        weights = [weight / sum_weight for weight in weights]
        self.layers = dict(zip(list(layers)+['output'], weights))
        self.norm = self.get_norm(512)

    def get_model(self, model_name, layers=(), fix_pad=False, real_mean=None, real_std=None, pred_mean=[0.5, 0.5, 0.5], pred_std=[0.225, 0.225, 0.225]):
        return GANVGGFeatures(network=model_name, layers=layers, fix_pad=fix_pad, real_mean=real_mean, real_std=real_std, gan_mean=pred_mean, gan_std=pred_std, )

    def get_loss(self):
        return nn.L1Loss()

    def get_norm(self, num_channels):
        return nn.InstanceNorm2d(num_channels, affine=False, track_running_stats=False)  # Please don't touch eps in norm, it affects image gray content

    def forward(self, pred, target):
        self.norm.eval()
        pred = self.perception(pred, is_pred=True)
        with torch.no_grad():
            target = self.perception(target, is_pred=False)
        loss = 0
        for layer, weight in self.layers.items():
            pred_i = pred[layer]
            target_i = target[layer]
            if self.apply_norm:
                pred_i = self.norm(pred_i)
                target_i = self.norm(target_i)
            loss += self.base_loss(pred_i, target_i) * weight
        return loss

    def extra_repr(self) -> str:
        return f'loss: {self.base_loss}, layers_weights: {self.layers}'


class ContentMSELoss(ContentLoss):
    def get_loss(self):
        return nn.MSELoss()

class AnimeGANContentLoss(nn.Module):
    vgg_19_to_bn = {0: 0,
                    2: 3,
                    5: 7,
                    7: 10,
                    10: 14,
                    12: 17,
                    14: 20,
                    16: 23,
                    19: 27,
                    21: 30,
                    23: 33,
                    25: 36}
    # Layers from https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa
    # Layers mapped on vgg19
    def __init__(self, model_name='vgg19_bn', color_weight=1, layers=(), weight_scaler=2, apply_norm=False, fix_pad=False, real_mean=None, real_std=None, pred_mean=[0.5, 0.5, 0.5], pred_std=[0.225, 0.225, 0.225]):
        super(AnimeGANContentLoss, self).__init__()

        self.color_weight = color_weight
        self.pred_std = pred_std
        self.pred_mean = pred_mean
        self.apply_norm = apply_norm
        self.base_loss = nn.L1Loss()
        self.color_loss = nn.MSELoss()
        self.perception = self.get_model(model_name=model_name, layers=layers, fix_pad=fix_pad, real_mean=real_mean, real_std=real_std, pred_mean=pred_mean, pred_std=pred_std)

        weights = list(reversed([1 / (weight_scaler ** i) for i in range(len(layers) + 1)]))
        sum_weight = sum(weights)
        weights = [weight / sum_weight for weight in weights]
        self.layers = dict(zip(list(layers)+['output'], weights))
        self.norm = self.get_norm(512)
        self.register_buffer('rgb_weight', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def get_model(self, model_name, layers=(), fix_pad=False, real_mean=None, real_std=None, pred_mean=[0.5, 0.5, 0.5], pred_std=[0.225, 0.225, 0.225]):
        return GANVGGFeatures(network=model_name, layers=layers, fix_pad=fix_pad, real_mean=real_mean, real_std=real_std, gan_mean=pred_mean, gan_std=pred_std, )

    def get_loss(self):
        return nn.L1Loss()

    def get_norm(self, num_channels):
        return nn.InstanceNorm2d(num_channels, affine=False, track_running_stats=False, eps=1e-2)

    def rgb2gray(self, tensor):
        return torch.sum(tensor * self.rgb_weight, 1, keepdim=True).repeat(1,3,1,1)

    def forward(self, pred, target):
        input_pred, input_target = pred, target
        self.norm.eval()
        pred = self.perception(self.rgb2gray(pred), is_pred=True)
        with torch.no_grad():
            target = self.perception(self.rgb2gray(target), is_pred=False)
        loss = 0
        for layer, weight in self.layers.items():
            pred_i = pred[layer]
            target_i = target[layer]
            if self.apply_norm:
                pred_i = self.norm(pred_i)
                target_i = self.norm(target_i)
            loss += self.base_loss(pred_i, target_i) * weight

        color_loss = self.color_loss(input_pred, input_target)
        return loss + color_loss * self.color_weight

    def extra_repr(self) -> str:
        return f'loss: {self.base_loss}, layers_weights: {self.layers}'

#  https://arxiv.org/pdf/2104.05623.pdf
class SWAGContentLoss(ContentLoss):
    # constructor [3, 4, 6, 3]
    #  conv2_3: 2  -  2
    #  conv3_4: 6  -  3+3
    #  conv4_6: 12 -  3+4+5
    #  conv5_3: 15 -  3+4+6+2
    def __init__(self, model_name='resnet50',
                 layers=None,
                 apply_norm=True,
                 fix_pad=False,
                 real_mean=None,
                 real_std=None,
                 pred_mean=[0.5, 0.5, 0.5],
                 pred_std=[0.225, 0.225, 0.225]):
        super().__init__(model_name=model_name, layers=layers, apply_norm=apply_norm, fix_pad=fix_pad, real_mean=real_mean, real_std=real_std, pred_mean=pred_mean, pred_std=pred_std)
        weights = list(reversed([1 / (2 ** i) for i in range(len(layers))]))
        sum_weight = sum(weights)
        weights = [weight / sum_weight for weight in weights]
        self.layers = dict(zip(list(layers[:-1]) + ['output'], weights))

    def get_model(self, model_name,
                  layers=(),
                  fix_pad=False,
                  real_mean=None,
                  real_std=None,
                  pred_mean=[0.5, 0.5, 0.5],
                  pred_std=[0.225, 0.225, 0.225]):
        return GANResNetFeatures(network=model_name,
                                 layers=layers,
                                 fix_pad=fix_pad,
                                 real_mean=real_mean,
                                 real_std=real_std,
                                 gan_mean=pred_mean,
                                 gan_std=pred_std)

    def get_loss(self):
        return nn.MSELoss()

    def get_norm(self, num_channels):
        class SoftMaxNorm(nn.Module):
            def forward(self, x):
                N, C, H, W = x.shape
                return torch.softmax(x.view(N, C, H * W), 2).view(N, C, H, W) * H * W

        return SoftMaxNorm()
