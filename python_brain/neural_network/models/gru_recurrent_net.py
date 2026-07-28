"""
gru_recurrent_net.py
====================
Network J — Gated Recurrent Unit (GRU)

A more efficient alternative to LSTM that often converges faster on 
high-frequency time series data.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class MarketGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3) # BUY, SELL, HOLD
        )

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, InputDim)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
