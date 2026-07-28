"""
graph_isomorphism_net.py
========================
Network AP — Graph Isomorphism Network (GIN)

A state-of-the-art GNN architecture capable of distinguishing complex 
inter-asset dependency structures that simpler GNNs might miss.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, h, adj):
        # h: (N, InDim), adj: (N, N)
        neighbor_sum = torch.matmul(adj, h)
        out = self.mlp((1 + self.eps) * h + neighbor_sum)
        return out

class MarketGIN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.l1 = GINLayer(input_dim, hidden_dim)
        self.l2 = GINLayer(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x, adj):
        h = F.relu(self.l1(x, adj))
        h = F.relu(self.l2(h, adj))
        return self.head(h)
