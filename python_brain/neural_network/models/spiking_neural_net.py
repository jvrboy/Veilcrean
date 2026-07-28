"""
spiking_neural_net.py
=====================
Network X — Spiking Neural Network (SNN)

Mimics biological neurons that transmit information via discrete spikes. 
Highly efficient for processing high-frequency data streams.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class SpikingLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.threshold = 1.0

    def forward(self, x):
        v = self.lin(x)
        # Simplified Leaky Integrate-and-Fire (LIF)
        spike = (v > self.threshold).float()
        return spike

class MarketSNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.l1 = SpikingLinear(input_dim, 64)
        self.l2 = SpikingLinear(64, 32)
        self.head = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor):
        h = self.l1(x)
        h = self.l2(h)
        return self.head(h)
