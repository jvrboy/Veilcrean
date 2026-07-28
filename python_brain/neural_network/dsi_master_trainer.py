"""
dsi_master_trainer.py
=====================
Specialized trainer for mastering Drift Switch Indices.
Uses Synthetic data patterns to fine-tune the TradeDecisionNet.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from ..neural_network.trainer import Trainer
from ..config import NN_CFG

class DSIMasterTrainer(Trainer):
    """
    Enhanced Trainer that uses reinforcement signals from 
    Synthetic Drift/Switch patterns.
    """
    
    def train_on_drift_patterns(self, X: np.ndarray, y: np.ndarray):
        """
        Custom training loop that penalizes the model heavily 
        for missing 'Switch' events.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        optimizer = torch.optim.Adam(self.trade_net.parameters(), lr=NN_CFG.learning_rate)
        # Weight the 'HOLD' (2) less, and 'BUY'/'SELL' more for high-frequency DSI
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.2, 1.2, 0.8]).to(self.device))
        
        self.trade_net.train()
        for epoch in range(10): # Quick fine-tuning
            optimizer.zero_grad()
            logits, conf = self.trade_net(X_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
            
        return loss.item()
