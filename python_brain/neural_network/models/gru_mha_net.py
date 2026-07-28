"""
gru_mha_net.py
===============
Network AU — GRU with Multi-Head Attention

Combines Gated Recurrent Units with a Transformer-style attention head 
to focus on key temporal events.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUMHANet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=n_heads, batch_first=True)
        self.head = nn.Linear(hidden_dim * 2, 3)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, InputDim)
        h, _ = self.gru(x)
        # h: (Batch, SeqLen, Hid*2)
        
        attn_out, _ = self.attn(h, h, h)
        # Max pool over sequence
        pooled = torch.max(attn_out, dim=1)[0]
        return self.head(pooled)
