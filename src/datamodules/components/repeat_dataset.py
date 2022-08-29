from torch.utils import data

class RepeatDataset(data.Dataset):
    """A wrapper of repeated dataset.
    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.
    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times=None, length=None):
        assert times is not None and length is not None or times is None and length is not None, 'One of length or times must be None and one must be not None'
        self.length = length
        self.dataset = dataset
        self.times = times

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]


    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len if self.times is not None else self.length