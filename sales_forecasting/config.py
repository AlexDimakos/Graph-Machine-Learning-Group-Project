from dataclasses import dataclass
from pathlib import Path

import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# TODO: Make this a tmp save for early stopping
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
CONFIG_PATH = Path(__file__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# choose from ['lstm', 'gcnlstm', 'gatgcnlstm', 'gru', 'gcngru', 'gatgcngru']
MODEL = "gcngru"
# choose from ['plant', 'group', 'subgroup', 'storage']
EDGE_TYPE = "plant"


@dataclass
class TrainingConfig:
    # Cross-validation folds
    n_splits: int = 3

    # Model parameters
    window_size: int = 6
    hidden_size: int = 8
    K: int = 2
    dropout: float = 0.3
    num_layers: int = 2

    # Training loop parameters
    batch_size: int = 8
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0005

    # Evaluation parameters
    eval_every: int = 1
    patience: int = 10

    save_path: str = MODELS_DIR / f"{MODEL}.pt"


# MLFlow
USE_MLFLOW = True


@dataclass
class MLFlowConfig:
    tracking_uri = f"file:{MLFLOW_DIR}"
    # choose from ['testing', MODEL]
    experiment_name = MODEL if MODEL in ["lstm", "gru"] else f"{MODEL}_{EDGE_TYPE}"
    run_name = "Fixed windowed dataset"
