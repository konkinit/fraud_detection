from dataclasses import dataclass


@dataclass
class LAYERS_DIMS:
    INPUT_DIM: int
    HIDDEN_DIM: int
    CODE_DIM: int


@dataclass
class MODEL_FEATURES:
    LEARNING_RATE: float
    N_EPOCHS: int


@dataclass
class PLOTTING_FEATURES:
    X: str = 'epoch'
    Y: str = 'loss'
    COLOR: str = "split"
    HEIGHT: int = 500
    WIDTH: int = 800
