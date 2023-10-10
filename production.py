import os
import sys
from fastapi import FastAPI
from pandas import read_parquet

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import Model_Inference


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Autoencoder fraud detector app"}


@app.get("/customer_id/{customer}")
def fraud_scorer(
        customer: int,
        model: str
) -> dict:
    X = read_parquet(
        f"./data/{model}_raw.gzip")
    error, label = Model_Inference(model)._error_classification_eval(
        X.values[customer])
    return {
        "customer_id": customer,
        "autoencoder_model_error": error,
        "fraud_label": label
    }
