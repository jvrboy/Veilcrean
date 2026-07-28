"""
deep_residual_shrinkage_net.py
==============================
Network AF — Deep Residual Shrinkage Network (DRSN)

An improved ResNet that uses soft thresholding (Shrinkage) to 
automatically filter out noise from technical features.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShrinkageBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Soft thresholding
        scales = self.fc(x)
        thres = scales.mean(dim=1, keepdim=True)
        out = torch.sign(x) * torch.relu(torch.abs(x) - thres)
        return x + out

class MarketDRSN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.shrinkage = nn.Sequential(*[ShrinkageBlock(hidden_dim) for _ in range(4)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.shrinkage(h)
        return self.head(h)
