import os
import sys
from fastapi import FastAPI

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import Model_Inference
from src.utils import read_data_from_s3


app = FastAPI()


@app.get("/")
def read_root():
    return {
        "Hello": "Autoencoder fraud detector app",
        "To perform inference, the endpoint is": """<ip-adress>:8800/customer_id/<customer_id>?model=<model_id>""",
    }


@app.get("/customer_id/{customer}")
def fraud_scorer(customer: int, model: str) -> dict:
    X = read_data_from_s3("data/all_customers.gzip")
    error, label = Model_Inference(model)._error_classification_eval(
        X.query(f"Ids == {customer}").drop(columns=["Ids", "Y"], axis=1).values
    )
    return {
        "customer_id": customer,
        "autoencoder_model_error": error,
        "fraud_label": label
    }
