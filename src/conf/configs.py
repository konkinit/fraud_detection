from dataclasses import dataclass


@dataclass
class LAYERS_DIMS:
    ENCODER_HIDDEN_DIM: int = 32
    ENCODER_OUTPUT_DIM: int = 32
    DECODER_HIDDEN_DIM: int = 32


@dataclass
class MODEL_FEATURES:
    LEARNING_RATE: float = 1e-3
    N_EPOCHS: int = 100
