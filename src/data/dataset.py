import torch
from torch.utils.data import Dataset


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
