import torch
from torch import nn

from .augment_pipe import AugmentPipe
from .torch_utils.ops import grid_sample_gradfix, conv2d_gradfix

AUGPIPE_SPECS = {
    'blit': dict(xflip=1, rotate90=1, xint=1),
    'geom': dict(scale=1, rotate=1, aniso=1, xfrac=1),
    'color': dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
    'filter': dict(imgfilter=1),
    'noise': dict(noise=1),
    'cutout': dict(cutout=1),
    'bg': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
    'bgc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1,
                hue=1, saturation=1),
    'bgcf': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1,
                 hue=1, saturation=1, imgfilter=1),
    'bgcfn': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                  lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
    'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1,
                   lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
}

def get_augments(name=None, augment_p=0):
    if name is None:
        return nn.Identity()
    else:
        conv2d_gradfix.enabled = True  # Improves training speed.
        grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.
        augment_pipe = AugmentPipe(**AUGPIPE_SPECS[name])
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        return augment_pipe