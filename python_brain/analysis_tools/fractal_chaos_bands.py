"""
fractal_chaos_bands.py
======================
Tool 95 — Fractal Chaos Bands (FCB)

Identifies market state by filtering out insignificant price action and 
plotting bands based on fractal highs/lows.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class FractalChaosBandsTool(BaseTool):
    name = "fractal_chaos"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 10:
            return result

        # A simplified fractal band (local 5-bar high/low)
        high = df["high"]
        low = df["low"]
        
        upper = high.rolling(5, center=True).max()
        lower = low.rolling(5, center=True).min()
        
        last_close = df["close"].iloc[-1]
        
        score = 0.0
        if last_close > upper.iloc[-3]: score = 0.5 # Breakout
        elif last_close < lower.iloc[-3]: score = -0.5
        
        result.score = score
        result.features = {
            "fcb_width": float((upper.iloc[-3] - lower.iloc[-3]) / last_close)
        }
        return result
