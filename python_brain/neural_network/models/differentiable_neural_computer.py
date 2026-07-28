"""
differentiable_neural_computer.py
==================================
Network AM — Differentiable Neural Computer (DNC)

A network augmented with an external memory matrix, allowing the bot 
to store and retrieve complex historical setups.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketDNC(nn.Module):
    def __init__(self, input_dim: int, mem_size: int = 128, mem_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        
        self.controller = nn.LSTM(input_dim + mem_dim, 128, batch_first=True)
        # Interface head: produces memory interaction params
        self.interface = nn.Linear(128, mem_dim + mem_size) # simplified (read vector + attention)
        
        self.head = nn.Linear(128 + mem_dim, 3)
        self.memory = nn.Parameter(torch.zeros(mem_size, mem_dim), requires_grad=False)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, InputDim)
        batch_size = x.size(0)
        h = torch.zeros(1, batch_size, 128).to(x.device)
        c = torch.zeros(1, batch_size, 128).to(x.device)
        read_vec = torch.zeros(batch_size, self.mem_dim).to(x.device)
        
        for t in range(x.size(1)):
            ctrl_in = torch.cat([x[:, t, :], read_vec], dim=-1).unsqueeze(1)
            out, (h, c) = self.controller(ctrl_in, (h, c))
            
            # Read from memory (Simplified Attention)
            params = self.interface(out.squeeze(1))
            attn = F.softmax(params[:, self.mem_dim:], dim=-1)
            read_vec = torch.matmul(attn, self.memory)
            
        final_h = torch.cat([out.squeeze(1), read_vec], dim=-1)
        return self.head(final_h)
