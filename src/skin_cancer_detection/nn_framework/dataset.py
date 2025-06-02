from torch.utils.data import Dataset
from typing import Literal
import torch


class BatchedDataset(Dataset):
    def __init__(self, path, set_label: Literal['train', 'valid'], dtype=None):
        """
        Args:
            set_label (Literal['train', 'valid']): Specifies whether the data belongs to the training or validation set.
            path: Path to data.
        """
        self.path = path
        self.dtype = dtype

    def __len__(self):
        # TODO: Remove after getting real data
        return 42

    def __getitem__(self, idx):
        # TODO: Remove after getting real data
        X = torch.rand(3, 224, 224)  # Example image
        Y = torch.tensor(1)  # Example label (e.g., class index)
        return X, Y

    @staticmethod
    def collate(batch):
        return ...