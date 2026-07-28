"""
deep_belief_net.py
==================
Network U — Deep Belief Network (DBN)

A generative graphical model that consists of multiple layers of 
Restricted Boltzmann Machines (RBMs).
"""
from __future__ import annotations
import torch
import torch.nn as nn

class MarketDBN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Sigmoid()) # Traditionally Sigmoid for DBNs
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 3)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.head(h)
