import os
import sys
from fastapi import FastAPI

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


app = FastAPI()
