"""
fibonacci.py
============
Tool 15 — Fibonacci Confluence

Automatically identifies recent major swings and finds OTE (Optimal Trade Entry) zones.
OTE is usually the 61.8% - 78.6% retracement area.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class FibonacciTool(BaseTool):
    name = "fibonacci"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("H4")
        if df is None or len(df) < 100:
            return result

        # 1. Identify major swing high/low in last 100 bars
        recent = df.tail(100)
        sw_high = recent["high"].max()
        sw_low  = recent["low"].min()
        
        idx_high = recent["high"].idxmax()
        idx_low  = recent["low"].idxmin()
        
        price = ctx.get("price", recent["close"].iloc[-1])
        diff = sw_high - sw_low
        if diff == 0: return result

        # 2. Determine if we are in an uptrend (low before high) or downtrend
        score = 0.0
        if idx_low < idx_high: # Potential Long retracement
            retracement = (sw_high - price) / diff
            # OTE Zone: 0.62 to 0.79
            if 0.618 <= retracement <= 0.786:
                score = 0.8
            elif 0.5 <= retracement < 0.618:
                score = 0.4
        else: # Potential Short retracement
            retracement = (price - sw_low) / diff
            if 0.618 <= retracement <= 0.786:
                score = -0.8
            elif 0.5 <= retracement < 0.618:
                score = -0.4
                
        result.score = score
        result.confidence = 0.65
        result.features = {
            "fib_retracement": float(score),
            "fib_in_ote": float(abs(score) >= 0.8)
        }
        return result
