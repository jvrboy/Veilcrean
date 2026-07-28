"""
radial_basis_net.py
===================
Network V — Radial Basis Function (RBF) Network

Uses Gaussian RBFs as activation functions, providing localized learning 
and strong interpolation for complex market states.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class RBFLinear(nn.Module):
    def __init__(self, in_features, out_features, n_centers=10):
        super().__init__()
        self.centers = nn.Parameter(torch.Tensor(n_centers, in_features).uniform_(-1, 1))
        self.beta = nn.Parameter(torch.Tensor(n_centers).fill_(1.0))
        self.head = nn.Linear(n_centers, out_features)

    def forward(self, x):
        # x: (Batch, InFeatures)
        # centers: (Centers, InFeatures)
        dist = torch.cdist(x, self.centers) # (Batch, Centers)
        rbf = torch.exp(-self.beta * dist**2)
        return self.head(rbf)

class MarketRBFNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.rbf = RBFLinear(input_dim, 64, n_centers=32)
        self.head = nn.Linear(64, 3)

    def forward(self, x):
        h = torch.relu(self.rbf(x))
        return self.head(h)
