"""
sequential_memory_lstm.py
=========================
Network F — LSTM Sequential Memory Net

An RNN-based model designed to capture long-term temporal dependencies
in market data.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class SequentialLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # Buy, Sell, Hold
        )

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, Features)
        out, (hn, cn) = self.lstm(x)
        # We take the output of the last time step
        last_step = out[:, -1, :]
        return self.head(last_step)
