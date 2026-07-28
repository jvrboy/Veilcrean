"""
variational_recurrent_net.py
=============================
Network BA — Variational Recurrent Neural Network (VRNN)

Combines an LSTM with a Variational Autoencoder at each time step 
to model the uncertainty of the underlying market hidden state.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class VRNNCell(nn.Module):
    def __init__(self, in_dim, hid_dim, latent_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim
        
        # Encoder (q)
        self.encoder = nn.Linear(in_dim + hid_dim, latent_dim * 2)
        # Prior (p)
        self.prior = nn.Linear(hid_dim, latent_dim * 2)
        # Decoder (f)
        self.decoder = nn.Linear(latent_dim + hid_dim, in_dim)
        # Recurrent state update (g)
        self.rnn = nn.GRUCell(in_dim + latent_dim, hid_dim)

    def forward(self, x, h):
        # x: (B, InDim), h: (B, HidDim)
        enc_out = self.encoder(torch.cat([x, h], dim=-1))
        mu, logvar = enc_out.chunk(2, dim=-1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        
        h_next = self.rnn(torch.cat([x, z], dim=-1), h)
        return h_next, z, mu, logvar

class MarketVRNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.cell = VRNNCell(input_dim, 128, 32)
        self.head = nn.Linear(128, 3)

    def forward(self, x_seq: torch.Tensor):
        batch, seq_len, dim = x_seq.size()
        h = torch.zeros(batch, 128).to(x_seq.device)
        for t in range(seq_len):
            h, _, _, _ = self.cell(x_seq[:, t, :], h)
        return self.head(h)
