from torch.utils import data
import numpy as np

class CustomLenDataset(data.Dataset):
    def __init__(self, dataset_len):
        self.dataset_len = dataset_len


    def __getitem__(self, index):
        new_index = np.array([index], dtype=np.float32)
        return new_index

    def __len__(self):
        return self.dataset_len
