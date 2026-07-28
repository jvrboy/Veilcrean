"""
temporal_conv_net.py
====================
Network R — Temporal Convolutional Network (TCN)

A 1D convolutional architecture that uses dilated causal convolutions 
to capture long-range temporal patterns without the memory bottleneck of LSTMs.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

class MarketTCN(nn.Module):
    def __init__(self, input_dim: int, num_channels=[64, 64, 64]):
        super().__init__()
        layers = []
        in_c = 1 # We treat InputDim as the Sequence length or use 1 channel
        for i, out_c in enumerate(num_channels):
            dilation = 2**i
            layers.append(TCNBlock(in_c, out_c, dilation))
            in_c = out_c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels[-1], 3)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen)
        x = x.unsqueeze(1) # (Batch, 1, SeqLen)
        h = self.tcn(x)
        # Global pooling
        h = h.mean(dim=2)
        return self.head(h)
