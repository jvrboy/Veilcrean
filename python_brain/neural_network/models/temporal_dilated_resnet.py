"""
temporal_dilated_resnet.py
==========================
Network AQ — Temporal Dilated ResNet

Uses dilated 1D convolutions within residual blocks to learn features 
at multiple temporal scales without increasing parameter count.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        h = self.bn(self.relu(self.conv1(x)))
        h = self.bn(self.conv2(h))
        return self.relu(x + h)

class TemporalDilatedResNet(nn.Module):
    def __init__(self, input_dim: int, channels: int = 64, n_blocks: int = 4):
        super().__init__()
        self.input_layer = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList([DilatedResBlock(channels, 2**i) for i in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 3)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen)
        h = x.unsqueeze(1) # (B, 1, L)
        h = self.input_layer(h)
        for block in self.blocks:
            h = block(h)
        h = self.pool(h).squeeze(-1)
        return self.head(h)
