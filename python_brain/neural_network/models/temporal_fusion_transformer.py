"""
temporal_fusion_transformer.py
==============================
Network L — Temporal Fusion Transformer (TFT)

An advanced attention-based architecture that integrates multi-horizon 
time series forecasting with static and dynamic variables.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Gate Linear Unit (GLU)
        self.glu = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=n_heads, batch_first=True)
        # Output head
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, InputDim)
        gate = self.glu(x)
        h = x * gate
        
        # Self-Attention
        attn_out, _ = self.attn(h, h, h)
        h = h + attn_out # Residual
        
        # Mean pooling
        pooled = h.mean(dim=1)
        return self.head(pooled)
