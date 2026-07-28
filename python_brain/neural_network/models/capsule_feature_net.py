"""
capsule_feature_net.py
======================
Network S — Capsule Network (CapsNet)

Uses capsule layers to model the spatial relationships between different 
technical features, helping the AI understand "feature hierarchy."
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketCapsuleNet(nn.Module):
    def __init__(self, input_dim: int, n_capsules: int = 8, cap_dim: int = 16):
        super().__init__()
        # Initial feature extraction
        self.conv = nn.Linear(input_dim, 128)
        
        # Capsules
        self.capsules = nn.ModuleList([
            nn.Linear(128, cap_dim) for _ in range(n_capsules)
        ])
        
        self.head = nn.Linear(n_capsules * cap_dim, 3)

    def squash(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (x / (norm + 1e-9))

    def forward(self, x):
        h = F.relu(self.conv(x))
        
        # Route through capsules
        caps_out = []
        for cap in self.capsules:
            caps_out.append(self.squash(cap(h)))
            
        h_caps = torch.cat(caps_out, dim=-1)
        return self.head(h_caps)
