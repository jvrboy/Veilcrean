"""
trailing_manager.py
===================
Dynamic trailing stop logic. Can be ATR-based or structure-based.
"""
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd
import numpy as np

class TrailingManager:
    def __init__(self, atr_multiplier: float = 2.0):
        self.atr_multiplier = atr_multiplier

    def compute_trailing_stop(self, 
                               symbol: str, 
                               direction: str, 
                               current_price: float, 
                               entry_price: float,
                               buffers: Dict[str, pd.DataFrame]) -> Optional[float]:
        """
        Calculates a new SL based on ATR or recent swings.
        """
        # Get M15 or H1 for trailing
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return None

        # Calculate ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]

        dist = atr * self.atr_multiplier
        
        if direction == "BUY":
            # Only trail upwards
            new_sl = current_price - dist
            return new_sl
        else:
            # Only trail downwards
            new_sl = current_price + dist
            return new_sl
