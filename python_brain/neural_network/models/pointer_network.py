"""
pointer_network.py
==================
Network AI — Pointer Network

Learns to "Point" to the most significant historical events (swings/volume peaks)
in a sequence, allowing the bot to identify key support/resistance dynamically.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.v  = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, InputDim)
        enc_out, (hn, cn) = self.encoder(x)
        dec_out, _ = self.decoder(hn.transpose(0, 1), (hn, cn))
        
        # Attention over enc_out (The Pointers)
        # Score = v * tanh(W1*enc + W2*dec)
        enc_feat = self.w1(enc_out) # (B, L, H)
        dec_feat = self.w2(dec_out) # (B, 1, H)
        
        scores = self.v(torch.tanh(enc_feat + dec_feat)).squeeze(-1) # (B, L)
        pointers = F.softmax(scores, dim=-1)
        
        return pointers
