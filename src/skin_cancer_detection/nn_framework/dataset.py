from torch.utils.data import Dataset
from typing import Literal


class BatchedDataset(Dataset):
    def __init__(self, set_label: Literal['train', 'valid'], dtype):
        """
        Args:
            set_label (Literal['train', 'valid']): Specifies whether the data belongs to the training or validation set.r.
            dtype
        """
        self.path = ...
        self.dtype = ...

    def __len__(self):
        return ...

    def __getitem__(self, idx):
        X = ...
        Y = ...
        return X, Y

    @staticmethod
    def collate(batch):
        return ...