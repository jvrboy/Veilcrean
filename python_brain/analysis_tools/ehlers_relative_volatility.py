"""
ehlers_relative_volatility.py
==============================
Tool 112 — Ehlers Relative Volatility Index

An alternative to RVI that uses a modified smoothing technique to 
measure the relative volatility of price moves.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class EhlersRVITool(BaseTool):
    name = "ehlers_rvi"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 10:
            return result

        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        
        # (Close - Open) + 2*(Close[1] - Open[1]) + 2*(Close[2] - Open[2]) + (Close[3] - Open[3]) / 6
        def smooth(s1, s2):
            val = s1 - s2
            return (val + 2 * val.shift(1) + 2 * val.shift(2) + val.shift(3)) / 6
            
        num = smooth(close, open_)
        den = smooth(high, low)
        
        # Super-Smoother filter proxy
        rvi = num.rolling(10).mean() / (den.rolling(10).mean() + 1e-9)
        
        last_rvi = rvi.iloc[-1]
        
        result.score = float(np.tanh(last_rvi * 5))
        result.features = {"ehlers_rvi_val": float(last_rvi)}
        return result
