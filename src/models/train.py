import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader

# add parent directory of src to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.config as config
from src.data.dataset import LSTMDataset
from src.models.model import LSTMBaseline
from src.utils.experiments import start_run


def evaluate_model(model, dataset, batch_size=64):
    model.eval()
    model.to(config.DEVICE)
    criterion = nn.MSELoss()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(config.DEVICE)
            y_true = batch["y"].to(config.DEVICE)
            y_pred = model(x)

            loss = criterion(y_pred, y_true)
            total_loss += loss.item() * x.size(0)

            all_preds.append(y_pred.cpu())
            all_trues.append(y_true.cpu())

    avg_loss = total_loss / len(dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)
    rmse = torch.sqrt(torch.mean((all_preds - all_trues) ** 2))
    return avg_loss, rmse.item(), all_preds, all_trues


def train_lstm(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=64,
    num_epochs=100,
    lr=0.001,
    eval_every=5,
    patience=10,
    save_path="best_model.pt",
):
    model.to(config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = np.inf
    patience_counter = 0

    # Stats for plotting
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "best_val_loss": np.inf,
        "best_val_rmse": np.inf,
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(config.DEVICE)
            y = batch["y"].to(config.DEVICE)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / len(train_dataset)
        history["train_loss"].append(avg_train_loss)

        mlflow.log_metric(
            key="train_loss",
            value=avg_train_loss,
            step=epoch,
        )

        # Periodic evaluation
        if val_dataset is not None and epoch % eval_every == 0:
            val_loss, val_rmse, _, _ = evaluate_model(
                model, val_dataset, config.LSTMConfig.batch_size
            )
            history["val_loss"].append(val_loss)
            history["val_rmse"].append(val_rmse)
            print(
                f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}"
            )

            # Log to MLFlow
            mlflow.log_metric(
                key="val_loss",
                value=val_loss,
                step=epoch,
            )
            mlflow.log_metric(
                key="val_rmse",
                value=val_rmse,
                step=epoch,
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                history["best_val_loss"] = best_val_loss
                history["best_val_rmse"] = val_rmse
                print(f" --> Best model saved at epoch {epoch}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        else:
            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")

    # Load best model
    if val_dataset is not None:
        model.load_state_dict(torch.load(save_path))

    return model, history


def scale_per_product(X_train, y_train, X_val, y_val):
    # Prepare scaled containers
    X_train_scaled = np.zeros_like(X_train)
    y_train_scaled = np.zeros_like(y_train)
    X_val_scaled = np.zeros_like(X_val)
    y_val_scaled = np.zeros_like(y_val)

    scaler_X_per_product = []
    scaler_y_per_product = []

    T, N, F = X_train.shape
    for node_idx in range(N):
        # Fit a scaler for features of this product
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_node = X_train[:, node_idx, :]  # (T, F)
        y_node = y_train[:, node_idx].reshape(-1, 1)

        X_train_scaled[:, node_idx, :] = scaler_X.fit_transform(X_node)
        y_train_scaled[:, node_idx] = scaler_y.fit_transform(y_node).ravel()

        # Scale val data using same scaler
        X_val_scaled[:, node_idx, :] = scaler_X.transform(X_val[:, node_idx, :])
        y_val_scaled[:, node_idx] = scaler_y.transform(
            y_val[:, node_idx].reshape(-1, 1)
        ).ravel()

        scaler_X_per_product.append(scaler_X)
        scaler_y_per_product.append(scaler_y)

    return (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        scaler_X_per_product,
        scaler_y_per_product,
    )


def run_lstm_training_pipeline(X_train, y_train, X_val, y_val):
    X_train, y_train, X_val, y_val, scaler_X_per_product, scaler_y_per_product = (
        scale_per_product(X_train, y_train, X_val, y_val)
    )

    train_dataset = LSTMDataset(
        X_train, y_train, window_size=config.LSTMConfig.window_size
    )
    val_dataset = LSTMDataset(X_val, y_val, window_size=config.LSTMConfig.window_size)

    input_size = X_train.shape[2]  # number of features (F)
    model = LSTMBaseline(input_size, config.LSTMConfig.hidden_size).to(config.DEVICE)

    model, history = train_lstm(
        model,
        train_dataset,
        val_dataset=val_dataset,
        batch_size=config.LSTMConfig.batch_size,
        num_epochs=config.LSTMConfig.epochs,
        lr=config.LSTMConfig.lr,
        eval_every=config.LSTMConfig.eval_every,
        patience=config.LSTMConfig.patience,
        save_path=config.LSTMConfig.save_path,
    )

    return model, history


def cross_validation_training(X, y):
    tscv = TimeSeriesSplit(n_splits=config.LSTMConfig.n_splits)
    """
    {
        "fold_k": {
            model,
            best_val_loss,
            best_val_rmse
        }
    }
    """
    cv_results = {}

    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        with mlflow.start_run(run_name=f"fold_{i}", nested=True):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            model, history = run_lstm_training_pipeline(X_train, y_train, X_val, y_val)

            mlflow.log_metrics(
                {
                    "best_val_loss": history["best_val_loss"],
                    "best_val_rmse": history["best_val_rmse"],
                }
            )

            cv_results[f"fold_{i}"] = {
                "model": model,
                "best_val_loss": history["best_val_loss"],
                "best_val_rmse": history["best_val_rmse"],
            }

            mlflow.pytorch.log_model(pytorch_model=model, name=model.__class__.__name__)

    # TODO: add better processing of results
    # Convert cv_results to a DataFrame
    results_df = pd.DataFrame(cv_results)

    # Calculate mean and std of best_val_loss
    mean_rmse = results_df.loc["best_val_rmse"].mean()
    std_rmse = results_df.loc["best_val_rmse"].std()

    mlflow.log_metrics({"mean_rmse": mean_rmse, "std_rmse": std_rmse})

    print(f"Mean best validation rmse: {mean_rmse:.4f} ± {std_rmse:.4f}")


def main():
    X, y = (
        np.load(config.PROCESSED_DATA_DIR / "X.npy"),
        np.load(config.PROCESSED_DATA_DIR / "y.npy"),
    )

    with start_run(config.MLFlowConfig.run_name):
        cross_validation_training(X, y)


if __name__ == "__main__":
    main()
