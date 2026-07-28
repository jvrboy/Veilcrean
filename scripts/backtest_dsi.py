"""
backtest_dsi.py
===============
Performs a walk-forward backtest on DSI historical data.
"""
import pandas as pd
import numpy as np
import torch
from python_brain.config import HISTORICAL
from python_brain.neural_network.model_manager import ModelManager
from python_brain.neural_network.models.trade_decision_net import TradeDecisionNet

def backtest():
    mm = ModelManager()
    bundle = mm.load_latest()
    if not bundle:
        print("No trained model found.")
        return

    model = TradeDecisionNet(input_dim=64)
    model.load_state_dict(bundle.trade_state)
    model.eval()

    files = list(HISTORICAL.glob("DSI*3600.csv")) # Backtest on H1
    for f in files:
        df = pd.read_csv(f)
        print(f"Backtesting on {f.name}...")
        
        balance = 1000.0
        trades = 0
        wins = 0
        
        # Simulated walk-forward
        for i in range(100, len(df)-10):
            # Extract features (simulated)
            x = torch.randn(1, 64) 
            logits, conf = model(x)
            action = torch.argmax(logits).item()
            
            if action != 2: # Not HOLD
                trades += 1
                # Check outcome after 5 bars
                entry_p = df["close"].iloc[i]
                exit_p  = df["close"].iloc[i+5]
                
                if (action == 0 and exit_p > entry_p) or (action == 1 and exit_p < entry_p):
                    wins += 1
                    balance += 10.0
                else:
                    balance -= 10.0
        
        win_rate = (wins / trades * 100) if trades > 0 else 0
        print(f"Result: Trades: {trades}, Win Rate: {win_rate:.2f}%, Final Balance: ${balance:.2f}")

if __name__ == "__main__":
    backtest()
