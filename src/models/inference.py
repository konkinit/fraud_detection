import os
import sys
import numpy as np
from tinydb import Query, TinyDB
import torch
from typing import Tuple

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import FraudAutoEncoder
from src.conf import LAYERS_DIMS, TINYDB_FEATURES
from src.utils import get_device


DEVICE = get_device(1)


class Model_Inference:
    def __init__(
            self,
            model_id: str,
    ):
        db = TinyDB(TINYDB_FEATURES.DB_PATH)
        table = db.table(TINYDB_FEATURES.TABLE)
        user = Query()
        model_metadata = table.search(
            user.model_id == model_id)[-1]
        self.fraud_cutoff = model_metadata["fraud_cutoff"]

        _LAYERS_DIMS = LAYERS_DIMS(
            INPUT_DIM=model_metadata["input_dim"],
            HIDDEN_DIM=model_metadata["hidden_dim"],
            CODE_DIM=model_metadata["code_dim"]
        )
        checkpoint = torch.load(
            f"./models/best_model_{model_id}.ckpt",
            map_location=DEVICE
        )
        self.model = FraudAutoEncoder(_LAYERS_DIMS).to(DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.error = torch.nn.MSELoss()

    def _error_classification_eval(
            self,
            x: torch.Tensor | np.ndarray
    ) -> Tuple[float, int]:
        """Evaluate reconstrction error and classify the obs
        as frauder or not

        Args:
            x (torch.Tensor | np.ndarray): input data

        Returns:
            Tuple[float, int]: tuple constitued of error and fraud label
        """
        _x = torch.from_numpy(
            x).to(DEVICE).to(torch.float32)

        with torch.no_grad():
            _error = float(self.error(_x, self.model(_x)).data)

        return _error, int(_error > self.fraud_cutoff)
