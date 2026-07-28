"""
keltner_width.py
================
Tool 87 — Keltner Channel Width

Measures the percentage spread of the Keltner Channels to identify
periods of volatility expansion and contraction.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class KeltnerWidthTool(BaseTool):
    name = "keltner_width"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        period = 20
        ma = df["close"].rolling(period).mean()
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        upper = ma + (1.5 * atr)
        lower = ma - (1.5 * atr)
        
        width = (upper - lower) / df["close"]
        last_width = width.iloc[-1]
        
        result.score = 0.0 # Volatility measure
        result.features = {"kcw_val": float(last_width)}
        return result
