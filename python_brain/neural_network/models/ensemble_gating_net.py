"""
ensemble_gating_net.py
======================
Network H — Ensemble Gating Network

A meta-learner that takes outputs from multiple models (MLP, CNN, LSTM)
and learns to weight them based on current market conditions.
"""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleGatingNet(nn.Module):
    def __init__(self, n_models: int = 3, feature_dim: int = 64):
        super().__init__()
        # Learns to produce weights for each model
        self.gating = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_models),
            nn.Softmax(dim=-1)
        )

    def forward(self, model_outputs: List[torch.Tensor], context_features: torch.Tensor):
        # model_outputs: list of Tensors shaped (Batch, 3)
        # context_features: (Batch, feature_dim)
        
        weights = self.gating(context_features) # (Batch, n_models)
        
        # Weighted sum of logits
        stacked_outputs = torch.stack(model_outputs, dim=1) # (Batch, n_models, 3)
        weighted_output = (stacked_outputs * weights.unsqueeze(-1)).sum(dim=1)
        
        return weighted_output
