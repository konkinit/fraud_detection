from dataclasses import dataclass


@dataclass
class LAYERS_FEATURES:
    ENCODER_HIDDEN_DIM: int = 32
    ENCODER_OUTPUT_DIM: int = 32
    DECODER_HIDDEN_DIM: int = 32


@dataclass
class MODEL_FEATURE:
    LEARNING_RATE: float = 1e-3
    N_EPOCHS: int = 200
