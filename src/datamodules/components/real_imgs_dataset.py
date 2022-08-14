import os.path
import random

import cv2
import numpy as np
from torch.utils import data
from scipy import signal

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    print(os.path.abspath(dir))
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class RealDataset(data.Dataset):
    """Dataset without order

    Note:
        dataroot param must point on folder with next structure
        dataroot:
          - trainA
          - trainB
          - testA
          - testB
          - valA
          - valB
        Some folders can not exists, depend on phases

    Args:
        transforms (Callable):
        dataroot (str): path to dataset
        phase (str): phase for folders
        direction (str): dataset
        input_nc (int): channels for A or B dataset (depend on direction)
        output_nc (int): channels for B or A dataset (depend on direction)
    """

    def __init__(self, transforms_real, dataroot, phase, real_folder='real', input_nc=3, output_nc=3):


        self.root = dataroot
        self.dir_real = os.path.join(dataroot, phase + '_' + real_folder)
        self.real_paths = make_dataset(self.dir_real)
        self.real_paths = sorted(self.real_paths)

        self.real_size = len(self.real_paths)

        self.transform_real = transforms_real

        self.input_nc = input_nc
        self.output_nc = output_nc

        self.phase = phase

    def __getitem__(self, index):
        real_path = self.real_paths[index % self.real_size]

        try:
            real_img = cv2.imread(real_path)
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        except:
            print('Error in ', real_path)

        real_base = self.transform_real.pre_transform(real_img)

        real = self.transform_real.strong_transform(real_base)

        real_base = self.transform_real.post_transform(real_base)
        real = self.transform_real.post_transform(real)


        if self.phase == 'val' or self.phase == 'test':
            return real

        input_nc = self.input_nc
        output_nc = self.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = real_img[0, ...] * 0.299 + real[1, ...] * 0.587 + real[2, ...] * 0.114
            real = tmp.unsqueeze(0)


        return dict(real=real, real_base=real_base, idx=index)

    def __len__(self):
        return self.real_size

    def name(self):
        return 'UnalignedDataset'
