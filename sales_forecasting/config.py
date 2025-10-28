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
# choose from ['lstm', 'gcnlstm', 'gatgcnlstm']
MODEL = "gatgcnlstm"
# choose from['plant', 'group', 'subgroup', 'storage']
EDGE_TYPE = "plant"


@dataclass
class TrainingConfig:
    # Cross-validation folds
    n_splits: int = 5

    # Model parameters
    window_size: int = 10
    hidden_size: int = 16
    K: int = 1

    # Training loop parameters
    batch_size: int = 16
    epochs: int = 100
    lr: float = 0.001

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
    experiment_name = MODEL
    # run_name = f"ws-{TrainingConfig.window_size}_lr-{TrainingConfig.lr}_hs{TrainingConfig.hidden_size}"
    run_name = "Initial run"
