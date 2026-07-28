"""
bidirectional_lstm_net.py
==========================
Network AC — Bidirectional LSTM

Processes the sequence of market data in both directions (past-to-present 
and current-to-past) to capture more robust temporal features.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class MarketBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        # Bidirectional means output is 2 * hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, InputDim)
        out, _ = self.lstm(x)
        # Global pooling across sequence
        pooled = out.mean(dim=1)
        return self.head(pooled)
