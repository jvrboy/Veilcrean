"""
neural_logic_net.py
===================
Network AO — Neural Logic Gate Network

A network that uses soft logic gates (AND, OR, NOT) to learn boolean 
decision rules directly from technical indicators.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftLogicGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(dim, dim).uniform_(-1, 1))

    def forward(self, x):
        # Soft AND: prod(x)
        # Soft OR: 1 - prod(1 - x)
        # We simulate with tanh/sigmoid logic
        return torch.sigmoid(torch.matmul(x, self.w))

class MarketLogicNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 64)
        self.gate1 = SoftLogicGate(64)
        self.gate2 = SoftLogicGate(64)
        self.head = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor):
        h = torch.sigmoid(self.l1(x))
        h = self.gate1(h)
        h = self.gate2(h)
        return self.head(h)
