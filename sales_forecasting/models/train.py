from dataclasses import asdict

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader

import sales_forecasting.config as config
from sales_forecasting.data.dataset import (
    LSTMDataset,
    WindowedStaticGraphTemporalSignal,
)
from sales_forecasting.models.model import (
    GATGCNLSTM,
    GCNGRUBaseline,
    GCNLSTMBaseline,
    LSTMBaseline,
)
from sales_forecasting.utils.experiments import start_run


def evaluate_model(model, dataset, batch_size=64, mask=None):
    """Evaluate model on either LSTMDataset (batched) or
    WindowedStaticGraphTemporalSignal (windowed GCN-LSTM).

    Returns: avg_loss, rmse, all_preds, all_trues
    """
    model.eval()
    model.to(config.DEVICE)
    criterion = nn.MSELoss()

    total_loss = 0.0
    all_preds, all_trues = [], []
    count = 0

    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.bool).to(config.DEVICE)

    is_lstm_dataset = isinstance(dataset, LSTMDataset)
    # Always use a DataLoader. For windowed datasets use batch_size=1.
    loader_bs = batch_size if is_lstm_dataset else 1
    loader = DataLoader(dataset, batch_size=loader_bs, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            if is_lstm_dataset:
                mask_batch = mask[batch["node_id"]]

                x = batch["x"].to(config.DEVICE)
                y_true = batch["y"].to(config.DEVICE)

                y_pred = model(x)

                y_pred = y_pred[mask_batch]
                y_true = y_true[mask_batch]

                if y_true.numel() == 0:
                    continue

                loss = criterion(y_pred, y_true)
                total_loss += loss.item() * x.size(0)
                count += x.size(0)

                all_preds.append(y_pred.cpu())
                all_trues.append(y_true.cpu())
            else:
                x = batch["x"].squeeze(0).to(config.DEVICE)
                edge_idx = batch["edge_index"].squeeze(0).to(config.DEVICE)
                edge_w = batch["edge_weight"].squeeze(0).to(config.DEVICE)
                y_true = batch["y"].squeeze(0).to(config.DEVICE)

                y_pred, _ = model(x, edge_idx, edge_w)

                y_pred = y_pred[mask]
                y_true = y_true[mask]

                loss = criterion(y_pred, y_true)
                total_loss += loss.item()
                count += 1

                all_preds.append(y_pred.cpu())
                all_trues.append(y_true.cpu())

    avg_loss = total_loss / count
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)
    rmse = torch.sqrt(torch.mean((all_preds - all_trues) ** 2))
    return avg_loss, rmse.item(), all_preds, all_trues


def train_model(
    model,
    train_dataset,
    train_config,
    val_dataset=None,
    batch_size=64,
    num_epochs=100,
    lr=0.001,
    eval_every=5,
    patience=10,
    weight_decay=5e-4,
    save_path="best_model.pt",
):
    mask = np.load(config.PROCESSED_DATA_DIR / "mask.npy")

    model.to(config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    is_lstm_dataset = isinstance(train_dataset, LSTMDataset)
    # Use DataLoader in both cases. For windowed dataset use batch_size=1 so
    # shuffled windows can be provided; we'll squeeze the fake batch dim.
    train_loader_bs = batch_size if is_lstm_dataset else 1
    train_loader = DataLoader(train_dataset, batch_size=train_loader_bs, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for batch in train_loader:
            if is_lstm_dataset:
                mask_batch = mask[batch["node_id"]]

                x = batch["x"].to(config.DEVICE)
                y = batch["y"].to(config.DEVICE)

                optimizer.zero_grad()
                y_pred = model(x)

                y_pred = y_pred[mask_batch]
                y = y[mask_batch]

                if y.numel() == 0:
                    continue

                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                count += x.size(0)
            else:
                x = batch["x"].squeeze(0).to(config.DEVICE)
                edge_idx = batch["edge_index"].squeeze(0).to(config.DEVICE)
                edge_w = batch["edge_weight"].squeeze(0).to(config.DEVICE)
                y = batch["y"].squeeze(0).to(config.DEVICE)

                optimizer.zero_grad()
                y_pred, _ = model(x, edge_idx, edge_w)

                y_pred = y_pred[mask]
                y = y[mask]

                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

        avg_train_loss = total_loss / count

        history["train_loss"].append(avg_train_loss)

        mlflow.log_metric(
            key="train_loss",
            value=avg_train_loss,
            step=epoch,
        )

        # Periodic evaluation
        if val_dataset is not None and epoch % eval_every == 0:
            val_loss, val_rmse, _, _ = evaluate_model(
                model, val_dataset, train_config.batch_size, mask=mask
            )
            history["val_loss"].append(val_loss)
            history["val_rmse"].append(val_rmse)
            print(
                f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}"
            )

            # Log to MLFlow
            mlflow.log_metrics(
                {
                    "val_loss": val_loss,
                    "val_rmse": val_rmse,
                },
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


def run_lstm_training_pipeline(X_train, y_train, X_val, y_val, train_config):
    X_train, y_train, X_val, y_val, scaler_X_per_product, scaler_y_per_product = (
        scale_per_product(X_train, y_train, X_val, y_val)
    )

    train_dataset = LSTMDataset(X_train, y_train, window_size=train_config.window_size)
    val_dataset = LSTMDataset(X_val, y_val, window_size=train_config.window_size)

    input_size = X_train.shape[2]  # number of features (F)
    model = LSTMBaseline(
        input_size,
        train_config.hidden_size,
        num_layers=train_config.num_layers,
        dropout=train_config.dropout,
    ).to(config.DEVICE)

    model, history = train_model(
        model,
        train_dataset,
        train_config=train_config,
        val_dataset=val_dataset,
        batch_size=train_config.batch_size,
        num_epochs=train_config.epochs,
        lr=train_config.lr,
        eval_every=train_config.eval_every,
        patience=train_config.patience,
        save_path=train_config.save_path,
        weight_decay=train_config.weight_decay,
    )

    return model, history, train_dataset, val_dataset


def run_gcnlstm_training_pipeline(X_train, y_train, X_val, y_val, train_config):
    X_train, y_train, X_val, y_val, scaler_X_per_product, scaler_y_per_product = (
        scale_per_product(X_train, y_train, X_val, y_val)
    )

    # Load correct edge_index
    edge_index = np.load(config.PROCESSED_DATA_DIR / f"edges_{config.EDGE_TYPE}.npy")
    train_dataset = WindowedStaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=np.ones(edge_index.shape[1]),
        features=X_train,
        targets=y_train,
    )
    val_dataset = WindowedStaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=np.ones(edge_index.shape[1]),
        features=X_val,
        targets=y_val,
    )

    input_size = X_train.shape[2]
    model = GCNLSTMBaseline(
        input_size=input_size,
        hidden_size=train_config.hidden_size,
        K=train_config.K,
    ).to(config.DEVICE)

    model, history = train_model(
        model,
        train_dataset,
        train_config=train_config,
        val_dataset=val_dataset,
        batch_size=train_config.batch_size,
        num_epochs=train_config.epochs,
        lr=train_config.lr,
        eval_every=train_config.eval_every,
        patience=train_config.patience,
        save_path=train_config.save_path,
        weight_decay=train_config.weight_decay,
    )

    return model, history, train_dataset, val_dataset


def run_gcngru_training_pipeline(X_train, y_train, X_val, y_val, train_config):
    X_train, y_train, X_val, y_val, scaler_X_per_product, scaler_y_per_product = (
        scale_per_product(X_train, y_train, X_val, y_val)
    )

    # Load correct edge_index
    edge_index = np.load(config.PROCESSED_DATA_DIR / f"edges_{config.EDGE_TYPE}.npy")
    train_dataset = WindowedStaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=np.ones(edge_index.shape[1]),
        features=X_train,
        targets=y_train,
    )
    val_dataset = WindowedStaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=np.ones(edge_index.shape[1]),
        features=X_val,
        targets=y_val,
    )

    input_size = X_train.shape[2]
    model = GCNGRUBaseline(
        input_size=input_size,
        hidden_size=train_config.hidden_size,
        K=train_config.K,
    ).to(config.DEVICE)

    model, history = train_model(
        model,
        train_dataset,
        train_config=train_config,
        val_dataset=val_dataset,
        batch_size=train_config.batch_size,
        num_epochs=train_config.epochs,
        lr=train_config.lr,
        eval_every=train_config.eval_every,
        patience=train_config.patience,
        save_path=train_config.save_path,
        weight_decay=train_config.weight_decay,
    )

    return model, history, train_dataset, val_dataset


def run_gatgcnlstm_training_pipeline(X_train, y_train, X_val, y_val, train_config):
    X_train, y_train, X_val, y_val, scaler_X_per_product, scaler_y_per_product = (
        scale_per_product(X_train, y_train, X_val, y_val)
    )

    # Load correct edge_index
    edge_index = np.load(config.PROCESSED_DATA_DIR / f"edges_{config.EDGE_TYPE}.npy")
    train_dataset = WindowedStaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=np.ones(edge_index.shape[1]),
        features=X_train,
        targets=y_train,
    )
    val_dataset = WindowedStaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=np.ones(edge_index.shape[1]),
        features=X_val,
        targets=y_val,
    )

    input_size = X_train.shape[2]
    model = GATGCNLSTM(
        input_size=input_size,
        hidden_size=train_config.hidden_size,
        K=train_config.K,
    ).to(config.DEVICE)

    model, history = train_model(
        model,
        train_dataset,
        train_config=train_config,
        val_dataset=val_dataset,
        batch_size=train_config.batch_size,
        num_epochs=train_config.epochs,
        lr=train_config.lr,
        eval_every=train_config.eval_every,
        patience=train_config.patience,
        save_path=train_config.save_path,
        weight_decay=train_config.weight_decay,
    )

    return model, history, train_dataset, val_dataset


def cross_validation_training(X, y, run_training_pipeline_fn, train_config):
    tscv = TimeSeriesSplit(n_splits=train_config.n_splits)
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

            model, history, train_dataset, val_dataset = run_training_pipeline_fn(
                X_train, y_train, X_val, y_val, train_config
            )

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
            mlflow.pytorch.log_model(
                pytorch_model=model.cpu(),
                name=model.__class__.__name__,
                # input_example=next(
                #     iter(
                #         DataLoader(
                #             train_dataset, batch_size=train_config.batch_size
                #         )
                #     )
                # )["x"]
                # .cpu()
                # .numpy(),
            )

    # TODO: add better processing of results
    # Convert cv_results to a DataFrame
    results_df = pd.DataFrame(cv_results)

    # Calculate mean and std of best_val_loss
    mean_rmse = results_df.loc["best_val_rmse"].mean()
    std_rmse = results_df.loc["best_val_rmse"].std()

    mlflow.log_metrics({"mean_rmse": mean_rmse, "std_rmse": std_rmse})

    print(f"Mean best validation rmse: {mean_rmse:.4f} ± {std_rmse:.4f}")


def run_experiment(train_config: config.TrainingConfig):
    X, y = (
        np.load(config.PROCESSED_DATA_DIR / "X.npy"),
        np.load(config.PROCESSED_DATA_DIR / "y.npy"),
    )

    if config.MODEL == "lstm":
        run_training_pipeline_fn = run_lstm_training_pipeline
    elif config.MODEL == "gcnlstm":
        run_training_pipeline_fn = run_gcnlstm_training_pipeline
    elif config.MODEL == "gcngru":
        run_training_pipeline_fn = run_gcngru_training_pipeline
    elif config.MODEL == "gatgcnlstm":
        run_training_pipeline_fn = run_gatgcnlstm_training_pipeline
    else:
        raise Exception(f"Unsupported model: {config.MODEL}")

    cross_validation_training(X, y, run_training_pipeline_fn, train_config)


if __name__ == "__main__":
    with start_run(config.MLFlowConfig.run_name):
        train_config = config.TrainingConfig()
        mlflow.log_params(asdict(train_config))
        run_experiment(train_config)
