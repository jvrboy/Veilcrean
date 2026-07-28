"""
volatility_ratio.py
===================
Tool 100 — Volatility Ratio (VR)

Identifies breakouts by comparing the current True Range to its 
average over a period.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class VolatilityRatioTool(BaseTool):
    name = "vol_ratio"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 14:
            return result

        n = 14
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(n).mean()
        vr = tr / (atr + 1e-9)
        
        last_vr = vr.iloc[-1]
        
        # VR > 2.0 suggests a significant breakout
        result.score = 0.0 # Indicator of momentum breakout
        result.features = {"vol_ratio_val": float(last_vr)}
        result.metadata = {"is_breakout": last_vr > 2.0}
        
        return result
