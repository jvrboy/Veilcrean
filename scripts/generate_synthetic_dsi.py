"""
generate_synthetic_dsi.py
=========================
Generates synthetic Drift Switch Index data for training when the API 
is unavailable.
"""
import pandas as pd
import numpy as np
import time
from python_brain.config import HISTORICAL

def generate_dsi(symbol: str, count: int = 10000):
    print(f"Generating synthetic data for {symbol}...")
    
    # Base parameters
    drift = 0.00001 if "10" in symbol else 0.00002 if "20" in symbol else 0.00005
    volatility = 0.001
    
    prices = [100.0]
    current_drift = drift
    
    for i in range(count):
        # Occasionally switch drift
        if np.random.random() < 0.01:
            current_drift *= -1
            
        change = np.random.normal(current_drift, volatility)
        prices.append(prices[-1] * (1 + change))
        
    df = pd.DataFrame({
        "epoch": np.arange(len(prices)),
        "open": prices,
        "high": np.array(prices) * 1.001,
        "low": np.array(prices) * 0.999,
        "close": prices
    })
    
    filename = HISTORICAL / f"{symbol}_3600.csv"
    df.to_csv(filename, index=False)
    print(f"Generated {len(df)} rows for {symbol}")

if __name__ == "__main__":
    for sym in ["DSI10", "DSI20", "DSI30"]:
        generate_dsi(sym)
