import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # predict scalar target

    def forward(self, x):
        # x: [B, window_size, F]
        out, _ = self.lstm(x)  # out: [B, window_size, hidden_size]
        out = out[:, -1, :]  # take output of last timestep: [B, hidden_size]
        out = self.fc(out)  # [B, 1]
        return out.squeeze(-1)  # [B]
