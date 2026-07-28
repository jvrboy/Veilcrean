"""
market_gan.py
=============
Network M — Market Generative Adversarial Network (GAN)

A GAN architecture designed to learn the distribution of market data 
and generate adversarial scenarios for stress-testing the main model.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class MarketGenerator(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class MarketDiscriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)
