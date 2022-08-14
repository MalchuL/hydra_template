import torchvision.transforms as transforms
from PIL import Image
import albumentations as A

# just modify the width and height to be multiple of 4
from albumentations import center_crop
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import Lambda
import cv2

from src.datamodules.transforms.staged_transform import StagedTransform


def get_infer_transform(max_size=384, must_divied=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    transform_list = []
    pre_process = [
        A.SmallestMaxSize(max_size, always_apply=True, interpolation=cv2.INTER_CUBIC),
        #A.CenterCrop(max_size, max_size, always_apply=True)
        ]

    if must_divied:
        class DividedResize(A.ImageOnlyTransform):
            def __init__(self, divided):
                super().__init__(always_apply=True)
                self.divided = divided

            def apply(self, img, **params):
                h, w, _ = img.shape
                h = h // must_divied * must_divied
                w = w // must_divied * must_divied

                img = center_crop(img, h, w)
                return img


        pre_process.append(DividedResize(must_divied))
    post_process = [A.Normalize(mean,
                                std),
                    ToTensorV2()]

    composed = pre_process + post_process

    composed = A.Compose(composed, p=1)

    transform_list += [Lambda(lambda x: composed(image=x)['image']),
                   ]
    return transforms.Compose(transform_list)

def get_cartoon_transform(load_size, fine_size, is_train, apply_strong=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    normal_prob = 0.3
    rare_prob = 0.1

    if is_train:
        pre_process = [
            A.ShiftScaleRotate(shift_limit=0.001, rotate_limit=20, scale_limit=0.3, interpolation=cv2.INTER_CUBIC,
                               p=normal_prob),
            A.SmallestMaxSize(load_size, always_apply=True, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(fine_size, fine_size, always_apply=True)]
    else:
        pre_process = [
            A.SmallestMaxSize(load_size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(fine_size, fine_size, always_apply=True)]

    if apply_strong:
        strong = []
    else:
        strong = [A.ToGray(p=rare_prob)]

    post_process = [A.Normalize(mean,
                                std),
                    ToTensorV2()]

    if not is_train:
        strong = []

    return StagedTransform(pre_process, strong, post_process)




def get_transform(load_size, fine_size, is_train, apply_strong=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    prob = 0.3

    if is_train:
        pre_process = [
            A.SmallestMaxSize(load_size, always_apply=True, interpolation=cv2.INTER_CUBIC),
            A.ShiftScaleRotate(shift_limit=0.001, rotate_limit=10, scale_limit=0.05, interpolation=cv2.INTER_CUBIC,
                               p=prob),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(fine_size, fine_size, always_apply=True),
        ]
    else:
        pre_process = [
            A.SmallestMaxSize(load_size, always_apply=True),
            A.RandomCrop(fine_size, fine_size, always_apply=True)]

    very_rare_prob = 0.05
    rare_prob = 0.1
    medium_prob = 0.2
    normal_prob = 0.3
    often_prob = 0.6
    compression_prob = 0.35
    if apply_strong:
        strong = []
    else:
        strong = [
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=normal_prob),
                A.MotionBlur(p=rare_prob),
                A.Downscale(scale_min=0.6, scale_max=0.8, interpolation=cv2.INTER_CUBIC, p=rare_prob),
            ], p=rare_prob),
            A.OneOf([
                A.ImageCompression(quality_lower=39, quality_upper=60, p=compression_prob),
                A.MultiplicativeNoise(multiplier=[0.92, 1.08], elementwise=True, per_channel=True, p=compression_prob),
                A.ISONoise(p=medium_prob)
            ], p=normal_prob),
            A.OneOf([
                A.ToGray(p=often_prob),
                A.ToSepia(p=medium_prob)
            ], p=rare_prob),
            A.OneOf([
                A.CLAHE(p=rare_prob),
                A.Equalize(by_channels=False, p=rare_prob),
            ], p=normal_prob),
            A.OneOf([
                A.RandomGamma(p=medium_prob),
                A.RGBShift(p=medium_prob),
            ], p=medium_prob),
            A.OneOf([
                A.HueSaturationValue(p=medium_prob),
                A.RandomBrightnessContrast(p=medium_prob, brightness_limit=(-0.3,0.2)),
            ], p=normal_prob),

        ]

    post_process = [A.Normalize(mean,
                                std),
                    ToTensorV2()]


    if not is_train:
        strong = []


    return StagedTransform(pre_process, strong, post_process)