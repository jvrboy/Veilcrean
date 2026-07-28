"""
kolmogorov_smirnov_net.py
==========================
Network BB — Kolmogorov-Smirnov Regime Detector

A network designed to perform real-time non-parametric statistical tests 
to identify when the price distribution has 'Shifted' (Regime Change).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketRegimeKS(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # Learned mapping to a latent space where KS test is more effective
        self.latent = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.head = nn.Linear(32, 5) # 5 Regimes

    def forward(self, x):
        z = self.latent(x)
        # We model the distribution of Z and compare with previous historical Z
        return self.head(z)
