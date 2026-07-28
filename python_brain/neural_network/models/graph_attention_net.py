"""
graph_attention_net.py
======================
Network AK — Graph Attention Network (GAT)

An advanced GNN that uses multi-head attention to learn which neighbor 
assets (correlated pairs) are most important for price prediction.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, h, adj):
        # h: (N, InDim), adj: (N, N)
        wh = self.w(h)
        n = h.size(0)
        
        # Self-attention mechanism
        # Repeat and concatenate features for all pairs
        a_input = torch.cat([wh.repeat(1, n).view(n * n, -1), wh.repeat(n, 1)], dim=1).view(n, n, -1)
        e = F.leaky_relu(self.a(a_input).squeeze(-1))
        
        # Masked attention based on adjacency (correlation)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        
        h_prime = torch.matmul(attention, wh)
        return h_prime

class MarketGAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gat1 = GATLayer(input_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, 3)

    def forward(self, x, adj):
        h = F.elu(self.gat1(x, adj))
        return self.gat2(h, adj)
