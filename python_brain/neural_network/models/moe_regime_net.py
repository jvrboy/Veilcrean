"""
moe_regime_net.py
=================
Network AB — Mixture of Experts (MoE)

Uses multiple "Expert" sub-networks for different market regimes, 
with a "Gating" network to select the best expert.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
    def forward(self, x): return self.net(x)

class MarketMoE(nn.Module):
    def __init__(self, input_dim: int, n_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(n_experts)])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor):
        weights = self.gate(x) # (Batch, n_experts)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1) # (Batch, n_experts, 3)
        
        # Weighted sum of expert opinions
        output = (expert_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return output
