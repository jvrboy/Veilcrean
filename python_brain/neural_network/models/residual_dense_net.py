"""
residual_dense_net.py
=====================
Network K — Residual Dense Network (ResNet)

Deep network using residual skip-connections to prevent vanishing gradients
and learn more complex hierarchical features.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return x + self.block(x) # Skip connection

class MarketResNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_blocks: int = 4):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.blocks(h)
        return self.head(h)
