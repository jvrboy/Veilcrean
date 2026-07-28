"""
chaikin_volatility.py
=====================
Tool 97 — Chaikin Volatility

Measures the difference between high and low prices to quantify volatility.
Expansion in range often precedes a trend reversal.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class ChaikinVolatilityTool(BaseTool):
    name = "chaikin_vol"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        n = 10
        m = 10
        hl = df["high"] - df["low"]
        ema_hl = hl.ewm(span=n, adjust=False).mean()
        
        cv = 100 * (ema_hl - ema_hl.shift(m)) / ema_hl.shift(m)
        
        last_cv = cv.iloc[-1]
        
        result.score = 0.0 # Volatility measure
        result.features = {"chaikin_vol_val": float(last_cv)}
        return result
