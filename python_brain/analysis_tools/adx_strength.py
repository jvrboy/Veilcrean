"""
adx_strength.py
===============
Tool 41 — Average Directional Index (ADX)

Measures the absolute strength of a trend, regardless of direction.
Useful for determining if trend-following or mean-reversion should be 
prioritized.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class ADXStrengthTool(BaseTool):
    name = "adx_strength"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 30:
            return result

        # Basic ADX implementation
        n = 14
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        
        # DM logic
        plus_dm.loc[plus_dm < minus_dm] = 0
        minus_dm.loc[minus_dm < plus_dm] = 0
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr_n = tr.rolling(n).mean()
        plus_di = 100 * (plus_dm.rolling(n).mean() / (atr_n + 1e-9))
        minus_di = 100 * (minus_dm.rolling(n).mean() / (atr_n + 1e-9))
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(n).mean()
        
        adx_val = adx.iloc[-1]
        
        result.score = 0.0 # Trend strength, not direction
        result.features = {
            "adx_val": float(adx_val),
            "plus_di": float(plus_di.iloc[-1]),
            "minus_di": float(minus_di.iloc[-1])
        }
        result.metadata = {"trend_strength": "STRONG" if adx_val > 25 else "WEAK"}
        
        return result
