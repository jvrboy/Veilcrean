"""
glu_mlp_net.py
==============
Network AN — Gated Linear Unit (GLU) MLP

Uses Gated Linear Units as activation functions to allow the network to 
dynamically block or pass information based on the input context.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GLULayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim * 2)
    def forward(self, x):
        h = self.lin(x)
        # GLU: a * sigmoid(b)
        a, b = h.chunk(2, dim=-1)
        return a * torch.sigmoid(b)

class MarketGLUNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.g1 = GLULayer(256)
        self.l2 = nn.Linear(256, 128)
        self.g2 = GLULayer(128)
        self.head = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.l1(x))
        h = self.g1(h)
        h = F.relu(self.l2(h))
        h = self.g2(h)
        return self.head(h)
