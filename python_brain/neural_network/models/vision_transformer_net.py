"""
vision_transformer_net.py
=========================
Network AL — Vision Transformer (ViT) for Price Action

Treats raw price action arrays as images and processes them through 
patch-based attention, allowing the bot to see "Macro Patterns."
"""
from __future__ import annotations
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 8, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (Batch, 1, SeqLen)
        x = self.proj(x) # (Batch, embed_dim, NumPatches)
        x = x.transpose(1, 2) # (Batch, NumPatches, embed_dim)
        return x

class MarketViT(nn.Module):
    def __init__(self, input_len: int = 64, patch_size: int = 8, embed_dim: int = 128, n_heads: int = 8):
        super().__init__()
        self.patch_embed = PatchEmbedding(1, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen)
        x = x.unsqueeze(1) # Add channel
        x = self.patch_embed(x)
        
        # Add CLS token
        b = x.size(0)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        h = self.transformer(x)
        return self.head(h[:, 0]) # Head of CLS token
