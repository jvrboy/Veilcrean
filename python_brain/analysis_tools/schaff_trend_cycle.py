"""
schaff_trend_cycle.py
=====================
Tool 59 — Schaff Trend Cycle (STC)

A hybrid indicator combining MACD with a Stochastic oscillator for a faster
and more accurate trend signal.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class STCTool(BaseTool):
    name = "stc"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 50:
            return result

        # Simplified STC logic
        close = df["close"]
        fast = 23
        slow = 50
        cycle = 10
        
        macd = close.ewm(span=fast).mean() - close.ewm(span=slow).mean()
        
        def stoch(s, period):
            return 100 * (s - s.rolling(period).min()) / (s.rolling(period).max() - s.rolling(period).min() + 1e-9)
            
        stc = stoch(stoch(macd, cycle), cycle) # Double smoothed stoch
        
        last_stc = stc.iloc[-1]
        
        result.score = float((last_stc - 50) / 50)
        result.features = {"stc_val": float(last_stc)}
        return result
