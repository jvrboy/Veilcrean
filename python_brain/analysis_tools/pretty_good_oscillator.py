"""
pretty_good_oscillator.py
=========================
Tool 69 — Pretty Good Oscillator (PGO)

Measures the distance between the current price and its n-day simple 
moving average, normalized by ATR.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class PGOTool(BaseTool):
    name = "pgo"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 14:
            return result

        n = 14
        close = df["close"]
        sma = close.rolling(n).mean()
        
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"] - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(n).mean()
        
        pgo = (close - sma) / (atr + 1e-9)
        
        last_pgo = pgo.iloc[-1]
        
        result.score = float(np.tanh(last_pgo / 3.0))
        result.features = {"pgo_val": float(last_pgo)}
        return result
