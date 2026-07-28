"""
feature_attention_mlp.py
========================
Network AD — Attention-augmented MLP

A dense network that uses self-attention on the input feature vector 
to learn which technical indicators are most relevant in real-time.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key   = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (Batch, input_dim)
        # We treat each feature as a 'token' of size 1
        x_tok = x.unsqueeze(-1) # (Batch, input_dim, 1)
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Attention score
        attn = torch.sigmoid(q * k) # Hadamard-based self-attention
        return x * attn # Gated output

class AttentionMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = FeatureAttention(input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        h = self.attn(x)
        return self.backbone(h)
