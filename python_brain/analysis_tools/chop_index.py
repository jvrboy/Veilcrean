"""
chop_index.py
=============
Tool 30 — Choppiness Index

Indicates whether the market is trending (low values) or ranging/choppy 
(high values).
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult

class ChopIndexTool(BaseTool):
    name = "chop_index"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 14:
            return result

        # Formula: 100 * LOG10( SUM(ATR(1), 14) / ( MaxHi(14) - MinLo(14) ) ) / LOG10(14)
        n = 14
        highs = df["high"].tail(n)
        lows = df["low"].tail(n)
        
        tr = pd.concat([
            highs - lows,
            (highs - df["close"].shift(1).tail(n)).abs(),
            (lows - df["close"].shift(1).tail(n)).abs()
        ], axis=1).max(axis=1)
        
        sum_tr = tr.sum()
        max_h = highs.max()
        min_l = lows.min()
        
        if max_h == min_l: return result
        
        chop = 100 * np.log10(sum_tr / (max_h - min_l)) / np.log10(n)
        
        result.score = 0.0 # Indicator of regime quality, not direction
        result.confidence = 1.0
        result.features = {"chop_index": float(chop)}
        result.metadata = {"chop_val": chop}
        
        return result
