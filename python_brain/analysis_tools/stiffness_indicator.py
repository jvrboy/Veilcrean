"""
stiffness_indicator.py
======================
Tool 92 — Stiffness Indicator

Measures the 'quality' of a trend by counting how many times price 
remains above a moving average over a period.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class StiffnessTool(BaseTool):
    name = "stiffness"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 100:
            return result

        n = 100
        m = 60 # SMA period
        close = df["close"]
        sma = close.rolling(m).mean()
        
        # Count bars where price > SMA
        above = (close > sma).rolling(n).sum()
        stiffness = (above / n) * 100
        
        last_stiff = stiffness.iloc[-1]
        
        result.score = float((last_stiff - 50) / 50)
        result.features = {"stiffness_val": float(last_stiff)}
        return result
