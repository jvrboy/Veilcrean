"""
risk_management_net.py
=======================
Network B — Risk Management Network

Input:  features + Network A's decision output (concatenated)
Output: (sl_distance, tp_distance, lot_size_multiplier)
        — all are normalized to [0, 1] and rescaled at inference
"""
from __future__ import annotations
import torch
import torch.nn as nn


class RiskManagementNet(nn.Module):
    def __init__(self, input_dim: int, n_actions: int = 3,
                 hidden_dims=(128, 64), dropout: float = 0.3):
        super().__init__()
        in_dim = input_dim + n_actions + 1   # +1 for confidence
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.sl_head = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())
        self.tp_head = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())
        self.lot_head= nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, action_onehot: torch.Tensor, confidence: torch.Tensor):
        h = torch.cat([x, action_onehot, confidence.unsqueeze(-1)], dim=-1)
        h = self.backbone(h)
        return self.sl_head(h), self.tp_head(h), self.lot_head(h)
