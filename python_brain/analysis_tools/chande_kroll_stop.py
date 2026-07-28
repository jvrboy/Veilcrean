"""
chande_kroll_stop.py
====================
Tool 117 — Chande Kroll Stop

A volatility-based trailing stop that uses ATR to project price boundaries 
and 'locks in' the highest/lowest values.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class ChandeKrollStopTool(BaseTool):
    name = "chande_kroll"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        p = 10
        x = 1
        q = 9
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        
        stop_long_pre = high.rolling(p).max() - (x * atr)
        stop_short_pre = low.rolling(p).min() + (x * atr)
        
        stop_long = stop_long_pre.rolling(q).min()
        stop_short = stop_short_pre.rolling(q).max()
        
        last_long = stop_long.iloc[-1]
        last_short = stop_short.iloc[-1]
        last_close = close.iloc[-1]
        
        result.score = 1.0 if last_close > last_short else -1.0 if last_close < last_long else 0.0
        result.features = {
            "ck_stop_long": float(last_long),
            "ck_stop_short": float(last_short)
        }
        return result
