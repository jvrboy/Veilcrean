"""
regime_classifier.py
====================
Network C — Market Regime Classifier

Input:  features
Output: 5-class softmax over (TRENDING, RANGING, VOLATILE, CHOPPY, BREAKOUT)
"""
from __future__ import annotations
import torch
import torch.nn as nn


REGIME_LABELS = ["TRENDING", "RANGING", "VOLATILE", "CHOPPY", "BREAKOUT"]


class RegimeClassifier(nn.Module):
    def __init__(self, input_dim: int, n_regimes: int = 5,
                 hidden_dims=(128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_regimes)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.head(h)   # logits
