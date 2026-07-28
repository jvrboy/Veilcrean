"""
liquid_state_machine.py
=======================
Network Z — Liquid State Machine (LSM)

A brain-inspired spiking reservoir model that excels at capturing 
high-frequency jitter and temporal patterns.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class SpikingLiquid(nn.Module):
    def __init__(self, input_dim: int, res_dim: int = 256):
        super().__init__()
        self.input_weights = nn.Linear(input_dim, res_dim)
        self.recurrent_weights = nn.Linear(res_dim, res_dim)
        self.res_dim = res_dim
        
        # Fixed (un-trained) reservoir
        for p in self.input_weights.parameters(): p.requires_grad = False
        for p in self.recurrent_weights.parameters(): p.requires_grad = False

    def forward(self, x, h):
        # x: current input
        # h: last spiking state
        v = self.input_weights(x) + self.recurrent_weights(h)
        spikes = (v > 0.5).float()
        return spikes

class MarketLSM(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.liquid = SpikingLiquid(input_dim, 256)
        self.readout = nn.Linear(256, 3)

    def forward(self, x_seq: torch.Tensor):
        # x_seq: (Batch, SeqLen, InputDim)
        batch_size = x_seq.size(0)
        h = torch.zeros(batch_size, 256).to(x_seq.device)
        
        for t in range(x_seq.size(1)):
            h = self.liquid(x_seq[:, t, :], h)
            
        return self.readout(h)
