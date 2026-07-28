"""
attn_recurrent_net.py
=====================
Network T — Attention-based Recurrent Neural Network

Combines GRU with an Attention mechanism to weight important hidden 
states in the sequence.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, InputDim)
        h, _ = self.gru(x)
        # h: (Batch, SeqLen, HiddenDim)
        
        attn_weights = self.attention(h) # (Batch, SeqLen, 1)
        context = torch.sum(h * attn_weights, dim=1) # (Batch, HiddenDim)
        
        return self.fc(context)
