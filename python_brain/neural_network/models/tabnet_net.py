"""
tabnet_net.py
=============
Network AV — TabNet (Attentive Interpretable Tabular Learning)

A model architecture designed for tabular data that uses sequential 
attention to choose which features to reason from at each decision step.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.glu = nn.GLU()

    def forward(self, x):
        h = self.bn(self.fc(x))
        # GLU divides dim by 2
        return h

class MarketTabNet(nn.Module):
    def __init__(self, input_dim: int, n_d: int = 64, n_a: int = 64):
        super().__init__()
        # Simplified TabNet logic
        self.attentive_transformer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.feature_transformer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_d + n_a)
        )
        self.head = nn.Linear(n_d, 3)

    def forward(self, x: torch.Tensor):
        # Mask features
        mask = self.attentive_transformer(x)
        h = x * mask
        # Transform
        h = self.feature_transformer(h)
        # Decision part
        d = h[:, :64]
        return self.head(d)
