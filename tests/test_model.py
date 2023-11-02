import os
import sys
import pytest
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import Model_Inference
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
    argnames=["model_id", "presence"],
    argvalues=[
        ("simulated_data", True),
        ("_simulated_data", False)
    ]
)
def test_model_presence(model_id: str, presence: bool):
    model_path = f"./models/best_model_{model_id}.ckpt"
    assert os.path.isfile(os.path.join(model_path)) == presence


@pytest.mark.parametrize(
    argnames=["model_id", "input"],
    argvalues=[
        ("simulated_data", data_for_testing[14]),
        ("simulated_data", data_for_testing[0])
    ]
)
def test_model_inference(
        model_id: str,
        input: Tensor | ndarray
):
    _model_inference = Model_Inference(model_id)
    error, label = _model_inference._error_classification_eval(input)
    assert label == int(error > _model_inference.fraud_cutoff)
