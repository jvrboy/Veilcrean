"""
volatility_pivot.py
===================
Tool 109 — Volatility Pivots

Dynamic support and resistance levels based on ATR multiples, providing 
reversal targets in volatile markets.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class VolatilityPivotTool(BaseTool):
    name = "vol_pivot"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        n = 14
        close = df["close"]
        high_low = df["high"] - df["low"]
        atr = high_low.rolling(n).mean()
        
        upper = close.rolling(n).mean() + (atr * 2)
        lower = close.rolling(n).mean() - (atr * 2)
        
        price = close.iloc[-1]
        dist = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-9)
        
        result.score = float((dist - 0.5) * 2)
        result.features = {"vol_pivot_pos": float(dist)}
        return result
