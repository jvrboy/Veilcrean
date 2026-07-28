"""
point_net.py
============
Network AJ — PointNet for Price Action

Treats price-volume clusters as a 3D point cloud, allowing the AI to 
capture spatial structures in market data regardless of time.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
    def forward(self, x): return self.mlp(x)

class MarketPointNet(nn.Module):
    def __init__(self, input_dim: int = 3): # (Price, Vol, Time)
        super().__init__()
        self.l1 = PointNetBlock(input_dim, 64)
        self.l2 = PointNetBlock(64, 128)
        self.l3 = PointNetBlock(128, 256)
        # Global pooling across points
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor):
        # x: (Batch, NumPoints, 3)
        b, n, d = x.size()
        h = self.l1(x.view(-1, d)).view(b, n, 64)
        h = self.l2(h.view(-1, 64)).view(b, n, 128)
        h = self.l3(h.view(-1, 128)).view(b, n, 256)
        
        # Max pool over points
        h_pool = torch.max(h, dim=1)[0]
        return self.head(h_pool)
