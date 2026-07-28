"""
normalized_macd.py
==================
Tool 114 — Normalized MACD

Standard MACD normalized by price or ATR, allowing the AI to compare 
momentum across different symbols on the same scale.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class NormalizedMACDTool(BaseTool):
    name = "norm_macd"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 26:
            return result

        close = df["close"]
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        
        macd = ema12 - ema26
        # Normalize by price
        norm_macd = macd / close * 100
        signal = norm_macd.ewm(span=9).mean()
        
        last_macd = norm_macd.iloc[-1]
        last_sig = signal.iloc[-1]
        
        result.score = float(np.tanh(last_macd - last_sig))
        result.features = {"norm_macd_val": float(last_macd)}
        return result
