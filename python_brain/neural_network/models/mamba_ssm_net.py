"""
mamba_ssm_net.py
================
Network AW — State Space Model (Mamba style)

A model architecture designed for extremely long sequences that uses 
Selective State Space Models (SSMs) to maintain a constant-size state.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class SSMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Simplified S6 logic: H = A*H + B*X
        self.A = nn.Parameter(torch.ones(dim))
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(dim, dim)

    def forward(self, x_seq):
        # x_seq: (B, L, D)
        batch, seq_len, dim = x_seq.size()
        h = torch.zeros(batch, dim).to(x_seq.device)
        states = []
        for t in range(seq_len):
            h = torch.sigmoid(self.A) * h + self.B(x_seq[:, t, :])
            states.append(self.C(h))
        return torch.stack(states, dim=1)

class MarketMambaNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.ssm = SSMBlock(hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x_seq: torch.Tensor):
        h = self.embedding(x_seq)
        h = self.ssm(h)
        # Average pool over time
        pooled = h.mean(dim=1)
        return self.head(pooled)
