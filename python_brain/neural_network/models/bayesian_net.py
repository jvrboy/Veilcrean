"""
bayesian_net.py
===============
Network O — Bayesian Neural Network (BNN)

Provides uncertainty estimates for every trade. The bot can learn to 
only trade when 'Epistemic Uncertainty' is low.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.rho_w = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3))
        
    def forward(self, x):
        sigma_w = torch.log1p(torch.exp(self.rho_w))
        w = self.mu_w + sigma_w * torch.randn_like(sigma_w)
        return F.linear(x, w)

class MarketBayesianNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.b1 = BayesianLinear(input_dim, 128)
        self.b2 = BayesianLinear(128, 64)
        self.head = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.b1(x))
        h = F.relu(self.b2(h))
        return self.head(h)

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 10):
        samples = torch.stack([self.forward(x) for _ in range(n_samples)])
        mean = samples.mean(dim=0)
        std  = samples.std(dim=0)
        return mean, std
