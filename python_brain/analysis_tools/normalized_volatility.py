"""
normalized_volatility.py
========================
Tool 94 — Normalized Volatility

Calculates the ATR relative to price, allowing for direct comparison 
of volatility across different assets (Forex vs Crypto).
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class NormalizedVolatilityTool(BaseTool):
    name = "normalized_vol"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 14:
            return result

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        norm_vol = (atr / df["close"]) * 100
        
        last_vol = norm_vol.iloc[-1]
        
        result.score = 0.0 # Context indicator
        result.features = {"norm_vol_val": float(last_vol)}
        return result
