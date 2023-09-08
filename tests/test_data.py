import os
import sys
import pytest
from pandas import read_parquet

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data import CustomersDataset, DataLoaders


data_for_testing = read_parquet(
    "./data/simulated_raw_data_new_arrival.gzip"
).values
data_for_testing_type = type(data_for_testing[0])
split_fractions = [0.7, 0.2, 0.1]


@pytest.mark.parametrize(
    argnames=["_data", "_type"],
    argvalues=[
        (data_for_testing, data_for_testing_type)
    ]
)
def test_CustomersDataset(_data, _type) -> None:
    assert len(_data) == len(
        CustomersDataset(_data)
    )
    assert type(CustomersDataset(_data)[0]) == _type


@pytest.mark.parametrize(
    argnames=["_data", "_splits_fraction", "_dataloaders"],
    argvalues=[
        (data_for_testing, split_fractions, 3),
    ]
)
def test_DataLoaders(_data, _splits_fraction, _dataloaders) -> None:
    assert len(DataLoaders(
        _data, _splits_fraction
    ).get_dataloaders()) == _dataloaders
