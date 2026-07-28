"""
wavenet_price_net.py
=====================
Network AS — WaveNet (Dilated Causal Convolutions)

A generative-style 1D CNN that uses causal dilations to capture 
very long-range dependencies in high-frequency price data.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class WaveBlock(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.gate = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        # x: (B, C, L)
        h = torch.tanh(self.conv(x)) * torch.sigmoid(self.gate(x))
        return x + h # Residual

class MarketWaveNet(nn.Module):
    def __init__(self, input_dim: int, n_blocks: int = 4):
        super().__init__()
        self.input_conv = nn.Conv1d(1, 64, kernel_size=1)
        self.blocks = nn.ModuleList([WaveBlock(64, 2**i) for i in range(n_blocks)])
        self.head = nn.Linear(64, 3)

    def forward(self, x):
        # x: (Batch, SeqLen)
        h = self.input_conv(x.unsqueeze(1))
        for block in self.blocks:
            h = block(h)
        return self.head(h.mean(dim=2))
