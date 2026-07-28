"""
sparse_transformer_net.py
=========================
Network AS — Sparse Transformer (Informer Style)

A transformer model that uses sparse self-attention mechanisms to 
handle very long price action sequences efficiently.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        # Simplified ProbSparse attention logic:
        # Instead of full N^2 attention, we only attend to 'dominant' keys
        scores = torch.matmul(self.q_proj(q), self.k_proj(k).transpose(-2, -1))
        # Top-K sparse mask
        top_scores, _ = torch.topk(scores, k=min(scores.size(-1), 16), dim=-1)
        mask = scores < top_scores[..., -1].unsqueeze(-1)
        scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, self.v_proj(v))

class MarketSparseTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attn = ProbAttention(hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor):
        h = self.embedding(x)
        # Sequence: (Batch, SeqLen, Hidden)
        h = self.attn(h, h, h)
        pooled = h.mean(dim=1)
        return self.head(pooled)
