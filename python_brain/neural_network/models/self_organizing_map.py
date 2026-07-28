"""
self_organizing_map.py
======================
Network W — Self-Organizing Map (SOM)

An unsupervised neural network that clusters market states into a 2D 
grid, helping the bot identify distinct market regimes.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class MarketSOM(nn.Module):
    def __init__(self, input_dim: int, grid_size=(10, 10)):
        super().__init__()
        self.grid_size = grid_size
        # Map weights
        self.weights = nn.Parameter(torch.Tensor(grid_size[0], grid_size[1], input_dim).uniform_(-1, 1))

    def forward(self, x):
        # x: (Batch, InputDim)
        # Find Best Matching Unit (BMU)
        # This is a non-differentiable unsupervised step usually
        batch_size = x.size(0)
        x_expanded = x.view(batch_size, 1, 1, -1)
        dist = torch.norm(x_expanded - self.weights, dim=-1) # (Batch, 10, 10)
        
        # Flatten grid to find min dist
        flat_dist = dist.view(batch_size, -1)
        bmu_idx = torch.argmin(flat_dist, dim=-1)
        
        return bmu_idx # Returns the cluster index
