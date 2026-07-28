"""
ma_ribbon.py
============
Tool 84 — Moving Average Ribbon

Uses a series of 8-10 exponential moving averages to identify trend 
alignment and exhaustion.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class MARibbonTool(BaseTool):
    name = "ma_ribbon"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 50:
            return result

        periods = [5, 10, 15, 20, 25, 30, 35, 40]
        emas = [df["close"].ewm(span=p, adjust=False).mean().iloc[-1] for p in periods]
        
        # Scoring: +1 if all aligned (Price > 5 > 10... > 40)
        price = df["close"].iloc[-1]
        
        # Bullish alignment count
        bull_count = sum(1 for i in range(len(emas)-1) if emas[i] > emas[i+1])
        if price > emas[0]: bull_count += 1
        
        score = (bull_count / (len(emas))) - 0.5
        
        result.score = float(score * 2)
        result.features = {"ribbon_alignment": float(bull_count)}
        return result
