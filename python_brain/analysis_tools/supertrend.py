"""
supertrend.py
=============
Tool 31 — SuperTrend Indicator

Classic trend-following indicator combining ATR and Median Price.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class SuperTrendTool(BaseTool):
    name = "supertrend"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        # Basic SuperTrend implementation
        period = 10
        multiplier = 3.0
        
        hl2 = (df["high"] + df["low"]) / 2
        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = ranges.max(axis=1).rolling(period).mean()
        
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        # Simplified trend logic
        in_uptrend = df["close"].iloc[-1] > lowerband.iloc[-1]
        
        result.score = 1.0 if in_uptrend else -1.0
        result.confidence = 0.7
        result.features = {
            "supertrend_direction": float(1.0 if in_uptrend else -1.0),
            "dist_to_band": float(abs(df["close"].iloc[-1] - (lowerband.iloc[-1] if in_uptrend else upperband.iloc[-1])))
        }
        
        return result
