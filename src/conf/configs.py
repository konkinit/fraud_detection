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
class TINYDB_FEATURES:
    DB_PATH: str = './data/models_metadata/metadata_history.json'
    TABLE: str = 'metadata_history'


@dataclass
class PLOTTING_FEATURES:
    X: str = 'epoch'
    Y: str = 'loss'
    COLOR: str = "split"
    RECONSTRUCTION_ERROR: str = "reconstruction_error"
    HEIGHT: int = 500
    WIDTH: int = 800
