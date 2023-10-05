import os
import sys
from tinydb import Query, TinyDB
import torch
import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import FraudAutoEncoder
from src.conf import LAYERS_DIMS
from src.utils import get_device


DEVICE = get_device(1)


class Model_Inference:
    def __init__(
            self,
            model_id: str,
    ):
        db = TinyDB('./data/models_metadata/metadata_history.json')
        table = db.table('metadata_history')
        user = Query()
        model_metadata = table.search(
            user.model_id == model_id)[-1]

        _LAYERS_DIMS = LAYERS_DIMS(
            INPUT_DIM=model_metadata["input_dim"],
            HIDDEN_DIM=model_metadata["hidden_dim"],
            CODE_DIM=model_metadata["code_dim"]
        )
        checkpoint = torch.load(
            f"./models/best_model_{model_id}.ckpt", map_location="cuda"
        )
        self.model = FraudAutoEncoder(_LAYERS_DIMS).to(DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.error = torch.nn.MSELoss()

    def _error_eval(self, x: torch.Tensor | np.ndarray) -> float:
        _x = torch.from_numpy(
            x).to(DEVICE).to(torch.float32)

        with torch.no_grad():
            return float(self.error(_x, self.model(_x)).data)

    def _fraud_classification(self):
        pass
