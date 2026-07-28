"""
trend_trigger_factor.py
========================
Tool 132 — Trend Trigger Factor (TTF)

Identifies trend reversals by comparing the highest high and lowest low 
of two adjacent time windows.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class TTFTool(BaseTool):
    name = "ttf"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 30: return res

        n = 15
        high = df["high"]
        low = df["low"]
        
        # Buy Power = HH(n) - LL(n) [current]
        # Sell Power = HH(n) - LL(n) [n periods ago]
        hh = high.rolling(n).max()
        ll = low.rolling(n).min()
        
        buy_power = hh - ll.shift(n)
        sell_power = hh.shift(n) - ll
        
        ttf = 100 * (buy_power - sell_power) / (0.5 * (buy_power + sell_power) + 1e-9)
        
        last_ttf = ttf.iloc[-1]
        
        # Thresholds: > 100 (Bullish), < -100 (Bearish)
        res.score = float(np.tanh(last_ttf / 100.0))
        res.features = {"ttf_val": float(last_ttf)}
        return res
