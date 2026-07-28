"""
reinforcement_head.py
=====================
Network Q — Deep Reinforcement Learning Head

Uses Policy Gradients (PPO/DQN style) to learn decision making 
based on cumulative reward (PnL) instead of just classification.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class RLPolicyHead(nn.Module):
    def __init__(self, backbone_dim: int, n_actions: int = 3):
        super().__init__()
        # Policy Network
        self.actor = nn.Sequential(
            nn.Linear(backbone_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        # Value Network (Critic)
        self.critic = nn.Sequential(
            nn.Linear(backbone_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, h: torch.Tensor):
        # h: Feature vector from backbone
        probs = self.actor(h)
        value = self.critic(h)
        return probs, value
