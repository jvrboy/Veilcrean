"""
volatility_stop.py
==================
Tool 145 — Volatility Stop (Ehlers style)

A trailing stop based on volatility (ATR) that uses a recursive 
averaging technique to identify high-probability exit points.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class VolatilityStopTool(BaseTool):
    name = "volatility_stop"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 30: return res

        n = 14
        tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(n).mean()
        
        # Volatility Stop
        v_stop_long = df["high"].rolling(n).max() - 2 * atr
        v_stop_short = df["low"].rolling(n).min() + 2 * atr
        
        last_c = df["close"].iloc[-1]
        
        res.score = 1.0 if last_c > v_stop_long.iloc[-1] else -1.0
        res.features = {
            "v_stop_pos": float((last_c - v_stop_long.iloc[-1]) / (2 * atr.iloc[-1] + 1e-9))
        }
        return res
