import os
import sys
from fastapi import FastAPI
from typing import Union

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/customer/{customer_id}")
def fraud_scorer(
        customer_id: int,
        score: Union[str, None] = None
) -> dict:
    return {"customer_id": customer_id, "score": score}
