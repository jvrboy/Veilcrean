"""
hierarchical_attention_net.py
==============================
Network AZ — Hierarchical Attention Network (HAN)

Uses dual levels of attention (Word/Feature level and Sentence/Sequence level) 
to understand hierarchical relationships in market features.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
    def forward(self, x):
        # x: (Batch, Len, Dim)
        weights = F.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)

class HierarchicalAttentionNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Feature-level attention
        self.feature_attn = nn.Linear(input_dim, input_dim)
        # Sequence-level attention
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.seq_attn = AttentionLayer(hidden_dim * 2)
        self.head = nn.Linear(hidden_dim * 2, 3)

    def forward(self, x_seq: torch.Tensor):
        # x_seq: (Batch, SeqLen, InputDim)
        # Apply feature weighting
        feat_w = torch.sigmoid(self.feature_attn(x_seq))
        h = x_seq * feat_w
        
        # Sequence modeling
        h, _ = self.gru(h)
        h = self.seq_attn(h)
        return self.head(h)
