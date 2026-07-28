"""
position_dqn.py
===============
Network AA — Deep Q-Network (DQN) for Position Management

Learns the optimal state-action value (Q) for managing open positions: 
Scale Out, Trail Stop, or Hold.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class PositionDQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int = 4):
        super().__init__()
        # Actions: 0: HOLD, 1: TRAIL, 2: SCALE_OUT, 3: FLATTEN
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, state: torch.Tensor):
        # state: account info + pnl + current tech features
        return self.q_net(state)
