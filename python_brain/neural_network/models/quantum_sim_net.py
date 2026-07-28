"""
quantum_sim_net.py
==================
Network BC — Quantum Simulation Network (Simulation)

A neural network architecture that simulates a variational quantum circuit 
(VQC) to perform interference-based pattern matching on technical features.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class MarketQuantumNet(nn.Module):
    def __init__(self, input_dim: int, n_qubits: int = 8):
        super().__init__()
        self.n_qubits = n_qubits
        # State preparation (Encoder)
        self.encoder = nn.Linear(input_dim, n_qubits)
        # Simulation of entanglement/rotation
        self.rotation = nn.Parameter(torch.randn(n_qubits))
        # Head
        self.head = nn.Linear(n_qubits, 3)

    def forward(self, x):
        # x: (Batch, InputDim)
        # 1. State preparation
        h = torch.tanh(self.encoder(x)) # Simulate 'bloch sphere' mapping
        
        # 2. Simulate Variational Circuit
        # We apply interference via element-wise multiplication with rotations
        h = h * torch.cos(self.rotation)
        
        # 3. Measurement (Linear Head)
        return self.head(h)
