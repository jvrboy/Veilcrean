"""
neural_arithmetic_logic_unit.py
================================
Network AG — Neural Arithmetic Logic Unit (NALU)

Designed to learn mathematical operations (addition, subtraction, 
multiplication) directly, helping the AI generalize numerical trends.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class NALU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_hat = nn.Parameter(torch.Tensor(out_dim, in_dim).uniform_(-1, 1))
        self.m_hat = nn.Parameter(torch.Tensor(out_dim, in_dim).uniform_(-1, 1))
        self.g = nn.Parameter(torch.Tensor(out_dim, in_dim).uniform_(-1, 1))

    def forward(self, x):
        w = torch.tanh(self.w_hat) * torch.sigmoid(self.m_hat)
        a = F.linear(x, w)
        m = torch.exp(F.linear(torch.log(torch.abs(x) + 1e-9), w))
        g = torch.sigmoid(F.linear(x, self.g))
        return g * a + (1 - g) * m

class MarketNALUNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.nalu1 = NALU(input_dim, 128)
        self.nalu2 = NALU(128, 64)
        self.head = nn.Linear(64, 3)

    def forward(self, x):
        h = self.nalu1(x)
        h = self.nalu2(h)
        return self.head(h)
