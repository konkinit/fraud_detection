import os
import sys
from torch import nn, Tensor
from typing import Any

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.conf import (
    LAYERS_DIMS,
)


class FraudAutoEncoder(nn.Module):
    def __init__(
            self,
            layer_dimensions: LAYERS_DIMS
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=layer_dimensions.INPUT_DIM,
                out_features=layer_dimensions.HIDDEN_DIM
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=layer_dimensions.HIDDEN_DIM,
                out_features=layer_dimensions.CODE_DIM,
            ),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=layer_dimensions.CODE_DIM,
                out_features=layer_dimensions.HIDDEN_DIM,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=layer_dimensions.HIDDEN_DIM,
                out_features=layer_dimensions.INPUT_DIM,
            ),
            nn.ReLU()
        )

    def forward(
            self,
            input_features: Tensor
    ) -> Any:
        """Forward pass

        Args:
            input_features (Tensor): input tensor

        Returns:
            Any: model output
        """
        _encoded = self.encoder(input_features)
        _decoded = self.decoder(_encoded)
        return _decoded
