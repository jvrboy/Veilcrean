"""
ltc_net.py
==========
Network AT — Liquid Time-Constant (LTC) Network

A brain-inspired recurrent network that models dynamic systems with 
time-varying time constants, ideal for irregular high-frequency ticks.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class LTCCell(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim + hid_dim, hid_dim)
        # Time-constant parameter
        self.tau = nn.Parameter(torch.ones(hid_dim))

    def forward(self, x, h):
        # x: (Batch, InDim), h: (Batch, HidDim)
        combined = torch.cat([x, h], dim=-1)
        h_dot = torch.tanh(self.lin(combined))
        # Continuous state update (Euler approx)
        h_next = h + (h_dot - h) / (1.0 + torch.exp(self.tau))
        return h_next

class MarketLTCNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.cell = LTCCell(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x_seq: torch.Tensor):
        # x_seq: (Batch, SeqLen, InputDim)
        batch_size = x_seq.size(0)
        h = torch.zeros(batch_size, 128).to(x_seq.device)
        
        for t in range(x_seq.size(1)):
            h = self.cell(x_seq[:, t, :], h)
            
        return self.head(h)
