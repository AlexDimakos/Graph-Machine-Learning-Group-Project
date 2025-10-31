import torch.nn as nn
import torch_geometric.nn
from torch_geometric_temporal.nn.recurrent import GConvGRU, GConvLSTM


class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)  # predict scalar target

    def forward(self, x):
        # x: [B, window_size, F]
        out, _ = self.lstm(x)  # out: [B, window_size, hidden_size]
        out = out[:, -1, :]  # take output of last timestep: [B, hidden_size]
        out = self.fc(out)  # [B, 1]
        return out.squeeze(-1)  # [B]


class GCNLSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, K=1):
        super().__init__()
        self.gcnlstm = GConvLSTM(in_channels=input_size, out_channels=hidden_size, K=K)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index, edge_weight, h=None, c=None):
        # x: [window_size, N, F]
        # edge_index: [window_size, 2, E]
        # edge_weight: [window_size, E]
        # h: [N, hidden_size]
        # c: [N, hidden_size]

        window_size, N, F = x.shape
        for t in range(window_size):
            # Pass previous hidden states (if any)
            h, c = self.gcnlstm(
                x[t, :, :], edge_index[t, :, :], edge_weight[t, :], h, c
            )
        out = self.linear(h)
        return out.squeeze(-1), (h, c)


class GATGCNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, K=1):
        super().__init__()
        self.gat = torch_geometric.nn.conv.GATv2Conv(
            in_channels=input_size,
            out_channels=input_size,
        )
        self.gcnlstm = GConvLSTM(in_channels=input_size, out_channels=hidden_size, K=K)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index, edge_weight, h=None, c=None):
        # x: [window_size, N, F]
        # edge_index: [window_size, 2, E]
        # edge_weight: [window_size, E]
        # h: [N, hidden_size]
        # c: [N, hidden_size]

        window_size, N, F = x.shape
        for t in range(window_size):
            # Apply GAT to get edge weights
            (_, (e_index, attention_weights)) = self.gat(
                x=x[t, :, :],
                edge_index=edge_index[t, :, :],
                return_attention_weights=True,
            )
            # Pass previous hidden states (if any)
            h, c = self.gcnlstm(
                x[t, :, :], e_index, attention_weights.squeeze(-1), h, c
            )
        out = self.linear(h)
        return out.squeeze(-1), (h, c)


class GRUBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # predict scalar target

    def forward(self, x):
        # x: [B, window_size, F]
        out, _ = self.gru(x)  # out: [B, window_size, hidden_size]
        out = out[:, -1, :]  # take output of last timestep: [B, hidden_size]
        out = self.fc(out)  # [B, 1]
        return out.squeeze(-1)  # [B]


class GCNGRUBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, K=1):
        super().__init__()
        self.gconvgru = GConvGRU(in_channels=input_size, out_channels=hidden_size, K=K)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index, edge_weight, h=None):
        # x: [window_size, N, F]
        window_size, N, F = x.shape
        for t in range(window_size):
            h = self.gconvgru(x[t, :, :], edge_index[t, :, :], edge_weight[t, :], h)
        out = self.linear(h)
        return out.squeeze(-1), h


class GATGCNGRU(nn.Module):
    def __init__(self, input_size, hidden_size, K=1):
        super().__init__()
        self.gat = torch_geometric.nn.conv.GATv2Conv(
            in_channels=input_size,
            out_channels=input_size,
        )
        self.gconvgru = GConvGRU(in_channels=input_size, out_channels=hidden_size, K=K)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index, edge_weight, h=None):
        # x: [window_size, N, F]
        window_size, N, F = x.shape
        for t in range(window_size):
            (_, (e_index, attention_weights)) = self.gat(
                x=x[t, :, :],
                edge_index=edge_index[t, :, :],
                return_attention_weights=True,
            )
            h = self.gconvgru(x[t, :, :], e_index, attention_weights.squeeze(-1), h)
        out = self.linear(h)
        return out.squeeze(-1), h
