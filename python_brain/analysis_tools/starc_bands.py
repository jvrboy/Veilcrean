"""
starc_bands.py
==============
Tool 148 — STARC Bands (Stoller Average Range Channels)

Bands created by adding and subtracting a multiple of ATR to a simple 
moving average.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class STARCBandsTool(BaseTool):
    name = "starc_bands"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20: return res

        n = 15
        close = df["close"]
        ma = close.rolling(n).mean()
        tr = pd.concat([df["high"] - df["low"], (df["high"] - close.shift(1)).abs(), (df["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(n).mean()
        
        upper = ma + (1.33 * atr)
        lower = ma - (1.33 * atr)
        
        last_c = close.iloc[-1]
        
        # Scoring: +1 if at lower band, -1 if at upper band
        score = (last_c - ma.iloc[-1]) / (1.33 * atr.iloc[-1] + 1e-9)
        
        res.score = -float(np.tanh(score))
        res.features = {"starc_pos": float(score)}
        return res
