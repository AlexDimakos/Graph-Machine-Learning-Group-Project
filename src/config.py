from dataclasses import dataclass
from pathlib import Path

import torch

# TODO: Track this via mlflow
# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# TODO: Make this a tmp save for early stopping
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LSTMConfig:
    n_splits = 2
    window_size = 10
    batch_size = 64
    hidden_size = 16
    epochs = 100
    lr = 0.001
    eval_every = 1
    patience = 10
    save_path = MODELS_DIR / "lstm.pt"


# MLFlow
USE_MLFLOW = True


@dataclass
class MLFlowConfig:
    tracking_uri = f"file:{MLFLOW_DIR}"
    experiment_name = "testing"  # "sales-forecasting"
    run_name = "test"
