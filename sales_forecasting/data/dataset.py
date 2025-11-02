import math

import torch
from torch.utils.data import Dataset
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class LSTMDataset(Dataset):
    def __init__(self, X, y, window_size):
        """
        Args:
            X: np.ndarray or torch.Tensor of shape [T, N, F]
            y: np.ndarray or torch.Tensor of shape [T, N]
            window_size: int, number of timesteps per input window
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)

        self.X = X
        self.y = y
        self.window_size = window_size
        self.T, self.N, self.F = X.shape

        # total number of possible windows per node
        self.num_windows_per_node = self.T - self.window_size

    def __len__(self):
        # Each node has (T - window_size) samples
        return self.num_windows_per_node * self.N

    def __getitem__(self, idx):
        """
        Given a flat index, map to (node_id, time_index)
        """
        node_id = idx // self.num_windows_per_node
        t_start = idx % self.num_windows_per_node
        t_end = t_start + self.window_size

        x = self.X[t_start:t_end, node_id, :]  # [window_size, F]
        y = self.y[t_end - 1, node_id]  # scalar target (last timestep)
        return {"x": x, "y": y, "node_id": node_id}


class WindowedStaticGraphTemporalSignal(StaticGraphTemporalSignal):
    def __init__(self, edge_index, edge_weight, features, targets, window_size=5):
        super().__init__(edge_index, edge_weight, features, targets)
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        num_snapshots = len(self.features)
        window_start = idx
        window_end = min(window_start + self.window_size, num_snapshots)
        actual_window_size = window_end - window_start

        _, N, F = self.features.shape
        windowed_x = torch.zeros((actual_window_size, N, F), dtype=torch.float32)
        windowed_edge_indices = torch.zeros(
            (actual_window_size,) + self.edge_index.shape, dtype=torch.long
        )
        windowed_edge_weights = torch.zeros(
            (actual_window_size,) + self.edge_weight.shape, dtype=torch.float32
        )
        windowed_y = torch.zeros(N, dtype=torch.float32)

        for i, timestep in enumerate(range(window_start, window_end)):
            snapshot = super().__getitem__(timestep)
            windowed_x[i, ...] = snapshot.x
            windowed_edge_indices[i, ...] = snapshot.edge_index
            windowed_edge_weights[i, ...] = snapshot.edge_attr
            windowed_y = snapshot.y

        return {
            "x": windowed_x,
            "edge_index": windowed_edge_indices,
            "edge_weight": windowed_edge_weights,
            "y": windowed_y,
        }

    def __next__(self):
        if self.t < len(self):
            window = self[self.t]
            self.t = self.t + 1
            return window
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self
