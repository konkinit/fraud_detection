import os
import sys
from torch import nn, Tensor, relu

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.conf import (
    LAYERS_FEATURES,
)


class FraudAutoEncoder(nn.Module):
    def __init__(
            self, input_dim: int, layer_features: LAYERS_FEATURES
    ) -> None:
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_dim,
            out_features=layer_features.ENCODER_HIDDEN_DIM
        )
        self.encoder_output_layer = nn.Linear(
            in_features=layer_features.ENCODER_HIDDEN_DIM,
            out_features=layer_features.ENCODER_OUTPUT_DIM,
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=layer_features.ENCODER_OUTPUT_DIM,
            out_features=layer_features.DECODER_HIDDEN_DIM,
        )
        self.decoder_output_layer = nn.Linear(
            in_features=layer_features.DECODER_HIDDEN_DIM,
            out_features=input_dim,
        )

    def forward(self, input_features: Tensor):
        activation = self.encoder_hidden_layer(input_features)
        activation = relu(activation)
        encode = self.encoder_output_layer(activation)
        encode = relu(encode)
        activation = self.decoder_hidden_layer(encode)
        activation = relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = relu(activation)
        return reconstructed
