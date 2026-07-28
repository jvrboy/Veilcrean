"""
transformer_attention_net.py
============================
Network G — Transformer Attention Network

Uses Multi-Head Attention to focus on the most important technical 
signals across different timeframes and indicators.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class MarketTransformer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, Features)
        e = self.embedding(x)
        h = self.transformer(e)
        # Global average pooling over the sequence
        pooled = h.mean(dim=1)
        return self.head(pooled)
