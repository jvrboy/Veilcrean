"""
ttm_squeeze.py
==============
Tool 75 — TTM Squeeze

Identifies periods of volatility compression (Squeeze) and the subsequent 
expansion (Release).
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class TTMSqueezeTool(BaseTool):
    name = "ttm_squeeze"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        period = 20
        close = df["close"]
        ma = close.rolling(period).mean()
        
        # Bollinger Bands
        std = close.rolling(period).std()
        upper_bb = ma + (2 * std)
        lower_bb = ma - (2 * std)
        
        # Keltner Channels
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"] - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        upper_kc = ma + (1.5 * atr)
        lower_kc = ma - (1.5 * atr)
        
        # Squeeze detection
        is_squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        
        # Momentum Histogram (Linear Regression of price from median)
        # Simplified:
        median_price = (df["high"].rolling(period).max() + df["low"].rolling(period).min()) / 2
        avg_median = (median_price + ma) / 2
        momentum = close - avg_median
        
        last_mom = momentum.iloc[-1]
        
        result.score = float(np.tanh(last_mom / (atr.iloc[-1] + 1e-9)))
        result.features = {
            "in_squeeze": float(is_squeeze.iloc[-1]),
            "squeeze_mom": result.score
        }
        return result
