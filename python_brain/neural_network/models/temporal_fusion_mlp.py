"""
temporal_fusion_mlp.py
=======================
Network AY — Temporal Fusion MLP

Uses parallel MLP branches to process technical features from different 
timeframes (M1, M15, H1) and fuses them into a single cross-TF reasoning.
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn

class TFBranch(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim)
        )
    def forward(self, x): return self.net(x)

class TemporalFusionMLP(nn.Module):
    def __init__(self, tf_dims: Dict[str, int], hidden_dim: int = 64):
        super().__init__()
        # Parallel branches for each TF (M1, M5, M15, etc)
        self.branches = nn.ModuleDict({
            tf: TFBranch(dim, hidden_dim) for tf, dim in tf_dims.items()
        })
        
        fusion_dim = hidden_dim * len(tf_dims)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, tf_inputs: Dict[str, torch.Tensor]):
        # tf_inputs: {'M1': x1, 'M15': x2, ...}
        branch_outs = []
        for tf, x in tf_inputs.items():
            branch_outs.append(self.branches[tf](x))
            
        fused = torch.cat(branch_outs, dim=-1)
        return self.head(fused)
