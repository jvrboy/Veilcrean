"""
residual_gru_net.py
====================
Network AR — Residual Gated Recurrent Unit (Res-GRU)

Combines the sequential memory of GRUs with residual skip-connections 
to allow for deeper recurrent modeling of time-series data.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class ResGRUBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.norm(x + h) # Residual addition

class MarketResGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_blocks: int = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResGRUBlock(hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor):
        h = self.embedding(x)
        h = self.blocks(h)
        # Pooling over sequence length
        pooled = h.mean(dim=1)
        return self.head(pooled)
