"""
pattern_recognition_cnn.py
==========================
Network E — CNN Pattern Recognition

Uses 1D convolutions to extract temporal patterns from price history.
Input: Sequence of price returns
Output: Feature vector for the main decision net
"""
from __future__ import annotations
import torch
import torch.nn as nn

class PatternCNN(nn.Module):
    def __init__(self, seq_len: int = 50, out_dim: int = 32):
        super().__init__()
        # Input: (Batch, 1, SeqLen)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor):
        # x shape expected: (Batch, SeqLen)
        x = x.unsqueeze(1) # Add channel dim
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)
