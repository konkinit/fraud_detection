import os
import sys
from numpy import array, ndarray
from torch import Tensor, float32, from_numpy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import List, Tuple


if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import ml_partitions_indices


dataloader_kwargs = {
    "batch_size": 16,
    "shuffle": True
}


class CustomersDataset(Dataset):
    def __init__(
            self,
            dataset,
            transform=None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return self.dataset.shape[0]

    def __getitem__(self, index):
        _data = self.dataset[index]
        if self.transform:
            _data = self.transform(_data)
        return _data


class DataLoaders:
    def __init__(
            self,
            raw_dataset: ndarray | Tensor,
            split_fractions: List[float]
    ) -> None:
        assert array(split_fractions).sum().round(1) == 1.0
        if raw_dataset is ndarray:
            raw_dataset = from_numpy(raw_dataset)
        raw_dataset = raw_dataset.to(float32)
        idx_arrays = ml_partitions_indices(
            raw_dataset.shape[0], split_fractions
        )
        self.train_dataset = CustomersDataset(raw_dataset[idx_arrays[0]])
        self.validation_dataset = CustomersDataset(raw_dataset[idx_arrays[1]])
        self.test_dataset = CustomersDataset(raw_dataset[idx_arrays[2]])

    def get_dataloaders(self) -> Tuple[DataLoader]:
        """Get train, validation and test Pytorch dataloaders

        Returns:
            Tuple[DataLoader]: dataloaders
        """
        return (
            DataLoader(self.train_dataset, **dataloader_kwargs),
            DataLoader(self.validation_dataset, **dataloader_kwargs),
            DataLoader(self.test_dataset, **dataloader_kwargs)
        )
