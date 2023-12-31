import os
import sys
import pytest
from pandas import DataFrame

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data import CustomersDataset, DataLoaders
from src.utils import read_data_from_s3


data_path = "test_data/typical_customers.gzip"


def test_data_presence():
    assert isinstance(read_data_from_s3(data_path), DataFrame)


df_for_testing = read_data_from_s3(data_path)
data_for_testing = df_for_testing.drop(columns=["Ids", "Y"], axis=1).values
ids_data_for_testing = df_for_testing["Ids"].values
data_for_testing_type = type(data_for_testing[0])
split_fractions = [0.7, 0.2, 0.1]


@pytest.mark.parametrize(
    argnames=["_data", "_type"],
    argvalues=[
        (data_for_testing, data_for_testing_type)
    ]
)
def test_CustomersDataset(_data, _type) -> None:
    _customer_data = CustomersDataset(_data, ids_data_for_testing)
    assert len(_data) == len(_customer_data)
    assert isinstance(_customer_data[0][1], _type)


@pytest.mark.parametrize(
    argnames=["_data", "_splits_fraction", "_dataloaders"],
    argvalues=[
        (df_for_testing, split_fractions, 3),
    ]
)
def test_DataLoaders(_data, _splits_fraction, _dataloaders) -> None:
    assert len(DataLoaders(
        _data, _splits_fraction
    ).get_dataloaders()) == _dataloaders
