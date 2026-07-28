"""
graph_correlation_net.py
========================
Network P — Graph Neural Network (GNN)

Models inter-asset correlations as a graph. Helps the bot understand 
contagion and rotation between Crypto, Forex, and Indices.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        # Message passing layers
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor):
        # node_features: (NumNodes, InputDim)
        # adj_matrix: (NumNodes, NumNodes) - Correlation based
        
        h = F.relu(self.node_embedding(node_features))
        
        # Simple GCN-like layer: H = Relu(Adj * H * W)
        h = F.relu(torch.matmul(adj_matrix, h))
        h = self.conv1(h)
        
        h = F.relu(torch.matmul(adj_matrix, h))
        h = self.conv2(h)
        
        return self.head(h)
