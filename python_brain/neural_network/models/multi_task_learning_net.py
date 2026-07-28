"""
multi_task_learning_net.py
==========================
Network AX — Multi-Task Learning (MTL) Network

A shared backbone with multiple 'Heads' that simultaneously predict 
Direction, Volatility, and Market Regime. This improves generalization 
by forcing the network to learn more robust features.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Task Heads
        self.direction_head = nn.Linear(hidden_dim // 2, 3) # BUY, SELL, HOLD
        self.volatility_head = nn.Linear(hidden_dim // 2, 1) # Expected ATR
        self.regime_head = nn.Linear(hidden_dim // 2, 5) # 5 Regimes

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        logits = self.direction_head(h)
        vol = torch.softplus(self.volatility_head(h))
        regime = self.regime_head(h)
        return logits, vol, regime
