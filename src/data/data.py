import os
import sys
from numpy import array
from pandas import DataFrame
from torch import float32, from_numpy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import List, Tuple


if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import ml_partitions_indices, get_device


DEVICE = get_device(1)


dataloader_kwargs = {
    "batch_size": 16,
    "shuffle": True
}


class CustomersDataset(Dataset):
    def __init__(
            self,
            data,
            ids,
            transform=None
    ) -> None:
        super().__init__()
        self.dataset = data
        self.ids = ids
        self.transform = transform

    def __len__(self) -> int:
        return self.dataset.shape[0]

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample_id = self.ids[index]
        if self.transform:
            sample = self.transform(sample)
        return sample_id, sample


class DataLoaders:
    def __init__(
            self,
            raw_dataset: DataFrame,
            split_fractions: List[float]
    ) -> None:
        assert array(
            split_fractions
        ).sum().round(1) == 1.0, "Sum of fractions diffrent of 1.0"
        data = raw_dataset.values
        ids = raw_dataset.index.values

        data = from_numpy(data).to(DEVICE).to(float32)
        idx_arrays = ml_partitions_indices(
            raw_dataset.shape[0], split_fractions
        )
        self.train_dataset = CustomersDataset(
            data[idx_arrays[0]],
            ids[idx_arrays[0]]
        )
        self.validation_dataset = CustomersDataset(
            data[idx_arrays[1]],
            ids[idx_arrays[1]]
        )
        self.test_dataset = CustomersDataset(
            data[idx_arrays[2]],
            ids[idx_arrays[2]]
        )

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
