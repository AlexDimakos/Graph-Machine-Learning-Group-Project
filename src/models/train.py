import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(model, dataset, batch_size=64):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_true = batch["y"].to(device)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / len(train_dataset)
        history["train_loss"].append(avg_train_loss)

        # Periodic evaluation
        if val_dataset is not None and epoch % eval_every == 0:
            val_loss, val_rmse, _, _ = evaluate_model(
                model, val_dataset, batch_size, criterion, device
            )
            history["val_loss"].append(val_loss)
            history["val_rmse"].append(val_rmse)
            print(
                f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}"
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


# def train_and_evaluate():
#     tscv = TimeSeriesSplit(n_splits=5)
#     """
#     {
#         "k-fold": {
#             model,
#             best_val_loss,
#             best_val_rmse
#         }
#     }
#     """
#     cv_results = {}

#     for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
#         X_train = X[train_idx]
#         X_test = X[test_idx]
#         y_train = y[train_idx]
#         y_test = y[test_idx]

#         # Prepare scaled containers
#         X_train_scaled = np.zeros_like(X_train)
#         y_train_scaled = np.zeros_like(y_train)
#         X_test_scaled = np.zeros_like(X_test)
#         y_test_scaled = np.zeros_like(y_test)

#         scaler_X_per_product = []
#         scaler_y_per_product = []

#         T, N, F = X_train.shape
#         for node_idx in range(N):
#             # Fit a scaler for features of this product
#             scaler_X = StandardScaler()
#             scaler_y = StandardScaler()

#             X_node = X_train[:, node_idx, :]  # (T, F)
#             y_node = y_train[:, node_idx].reshape(-1, 1)

#             X_train_scaled[:, node_idx, :] = scaler_X.fit_transform(X_node)
#             y_train_scaled[:, node_idx] = scaler_y.fit_transform(y_node).ravel()

#             # Scale test data using same scaler
#             X_test_scaled[:, node_idx, :] = scaler_X.transform(X_test[:, node_idx, :])
#             y_test_scaled[:, node_idx] = scaler_y.transform(
#                 y_test[:, node_idx].reshape(-1, 1)
#             ).ravel()

#             scaler_X_per_product.append(scaler_X)
#             scaler_y_per_product.append(scaler_y)

#         window_size = int(
#             0.1 * T
#         )  # TODO: Consider here the implications of window size and the possibility of training a rolling LSTM with no fixed window size
#         batch_size = 64

#         train_dataset = GraphTimeSeriesDataset(
#             X_train_scaled, y_train_scaled, window_size=window_size
#         )
#         test_dataset = GraphTimeSeriesDataset(
#             X_test_scaled, y_test_scaled, window_size=window_size
#         )

#         train_dataloader = DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True
#         )
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#         input_size = F  # number of features per node
#         hidden_size = 16

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         model = LSTMBaseline(input_size, hidden_size).to(device)

#         model, history = train_lstm(
#             model,
#             train_dataset,
#             val_dataset=test_dataset,
#             window_size=5,
#             batch_size=batch_size,
#             num_epochs=100,
#             lr=0.001,
#             eval_every=1,
#             patience=20,
#         )
#         cv_results[f"fold_{i}"] = {
#             "model": model,
#             "best_val_loss": history["best_val_loss"],
#             "best_val_rmse": history["best_val_rmse"],
#         }
