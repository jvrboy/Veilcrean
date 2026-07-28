"""
echo_state_net.py
=================
Network Y — Echo State Network (ESN)

A type of reservoir computing where only the output weights are trained, 
providing extremely fast learning and stable time-series memory.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class MarketESN(nn.Module):
    def __init__(self, input_dim: int, reservoir_dim: int = 500):
        super().__init__()
        self.reservoir_dim = reservoir_dim
        # Fixed reservoir weights
        self.w_in = nn.Linear(input_dim, reservoir_dim)
        self.w_res = nn.Linear(reservoir_dim, reservoir_dim)
        # Non-trainable
        for param in self.w_in.parameters(): param.requires_grad = False
        for param in self.w_res.parameters(): param.requires_grad = False
        
        # Trainable readout head
        self.readout = nn.Linear(reservoir_dim, 3)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, InputDim)
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.reservoir_dim).to(x.device)
        
        # Sequentially update reservoir
        for t in range(x.size(1)):
            h = torch.tanh(self.w_in(x[:, t, :]) + self.w_res(h))
            
        return self.readout(h)
