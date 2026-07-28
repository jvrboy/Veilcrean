"""
donchian_width.py
=================
Tool 85 — Donchian Channel Width (DCW)

Measures the percentage width of Donchian Channels to identify volatility 
compression and breakouts.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class DonchianWidthTool(BaseTool):
    name = "donchian_width"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        period = 20
        upper = df["high"].rolling(period).max()
        lower = df["low"].rolling(period).min()
        
        width = (upper - lower) / df["close"]
        
        last_width = width.iloc[-1]
        
        result.score = 0.0 # Volatility measure
        result.features = {"dcw_val": float(last_width)}
        result.metadata = {"volatility_state": "EXPANDING" if last_width > width.rolling(20).mean().iloc[-1] else "SQUEEZING"}
        return result
