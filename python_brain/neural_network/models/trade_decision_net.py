"""
trade_decision_net.py
=====================
Network A — Trade Decision Network

Input:  feature vector of size N (filled dynamically)
Output: 3 logits (BUY, SELL, HOLD) + 1 confidence value in [0, 1]
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TradeDecisionNet(nn.Module):
    """A simple MLP head that outputs action logits + scalar confidence."""

    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64),
                 n_actions: int = 3, dropout: float = 0.3):
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
        self.action_head   = nn.Linear(prev, n_actions)
        self.confidence_head = nn.Sequential(
            nn.Linear(prev, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        logits  = self.action_head(h)
        conf    = self.confidence_head(h)
        return logits, conf.squeeze(-1)
