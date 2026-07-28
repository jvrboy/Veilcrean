"""
volatility_forecaster.py
========================
Network D — Volatility Forecasting Network

Predicts the expected range (ATR) of the next few bars to optimize SL/TP.
Input: Feature vector
Output: Scalar (Projected Volatility)
"""
from __future__ import annotations
import torch
import torch.nn as nn

class VolatilityForecasterNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h)
            ]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(prev, 1),
            nn.Softplus() # Ensure positive output
        )

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.head(h).squeeze(-1)
