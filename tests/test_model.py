import os
import sys
import pytest
from numpy import ndarray
from pandas import read_parquet
from torch import Tensor

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import Model_Inference


data_path = "./data/simulated_data_raw_new_arrival.gzip"


def test_data_presence():
    assert os.path.isfile(os.path.join(data_path))


df_for_testing = read_parquet(
    data_path
)
data_for_testing = df_for_testing.values


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
