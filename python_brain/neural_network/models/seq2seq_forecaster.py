"""
seq2seq_forecaster.py
=====================
Network AH — Sequence-to-Sequence (Seq2Seq) Forecaster

An Encoder-Decoder architecture designed to map a historical sequence 
of price action to a future sequence of predicted prices.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        return hn, cn

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, hn, cn):
        out, (hn, cn) = self.lstm(x, (hn, cn))
        prediction = self.fc(out)
        return prediction, hn, cn

class MarketSeq2Seq(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(3, hidden_dim) # Predict BUY, SELL, HOLD probabilities for N steps

    def forward(self, x: torch.Tensor, horizon: int = 5):
        # x: (Batch, SeqLen, InputDim)
        hn, cn = self.encoder(x)
        
        # Start decoding
        batch_size = x.size(0)
        input_token = torch.zeros(batch_size, 1, 3).to(x.device)
        predictions = []
        
        for _ in range(horizon):
            pred, hn, cn = self.decoder(input_token, hn, cn)
            predictions.append(pred)
            input_token = pred # Autoregressive feeding
            
        return torch.cat(predictions, dim=1)
