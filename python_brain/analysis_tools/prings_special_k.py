"""
prings_special_k.py
===================
Tool 88 — Pring's Special K

A complex momentum indicator that combines multiple timeframes of 
ROC and weighted moving averages to spot major trend shifts.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class SpecialKTool(BaseTool):
    name = "special_k"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H4") or buffers.get("H1")
        if df is None or len(df) < 100:
            return result

        # Simplified version of Special K
        close = df["close"]
        
        def sma_roc(s, r, m):
            roc = ((s - s.shift(r)) / s.shift(r)) * 100
            return roc.rolling(m).mean()
            
        # Composite of various ROCs
        k = (sma_roc(close, 10, 10) * 1 +
             sma_roc(close, 15, 10) * 2 +
             sma_roc(close, 20, 10) * 3 +
             sma_roc(close, 30, 15) * 4)
             
        last_k = k.iloc[-1]
        
        result.score = float(np.tanh(last_k / 50.0))
        result.features = {"special_k_val": float(last_k)}
        return result
