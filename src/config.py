from dataclasses import dataclass
from pathlib import Path

import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "outputs/models"
MLFLOW_DIR = PROJECT_ROOT / "outputs/mlflow"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LSTMConfig:
    n_splits = 5
    window_size = 10
    batch_size = 64
    hidden_size = 16
    epochs = 100
    lr = 0.001
    eval_every = 5
    patience = 10
    save_path = Path("outputs/models/lstm.pt")
