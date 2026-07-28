"""
ms_dropout_mlp.py
==================
Network AR — Multi-Sample Dropout MLP

Uses multiple dropout masks for the same hidden layer during training 
and inference to produce a more robust and stable ensemble effect.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class MSDropoutMLP(nn.Module):
    def __init__(self, input_dim: int, n_samples: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for _ in range(n_samples)])
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        # Multi-sample dropout
        outputs = [self.fc2(drop(h)) for drop in self.dropouts]
        return torch.stack(outputs).mean(dim=0)
