"""
relative_vigor_index.py
========================
Tool 82 — Relative Vigor Index (RVI)

Measures the conviction of a current price trend based on the relationship 
between closing and opening prices.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class RelativeVigorIndexTool(BaseTool):
    name = "relative_vigor"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 14:
            return result

        # (Close - Open) / (High - Low)
        num = (df["close"] - df["open"]).rolling(4).mean()
        den = (df["high"] - df["low"]).rolling(4).mean()
        
        rvi = num.rolling(10).mean() / (den.rolling(10).mean() + 1e-9)
        signal = rvi.rolling(4).mean()
        
        last_rvi = rvi.iloc[-1]
        last_sig = signal.iloc[-1]
        
        result.score = float(np.tanh(last_rvi - last_sig))
        result.features = {"rvi_vigor_val": float(last_rvi)}
        return result
