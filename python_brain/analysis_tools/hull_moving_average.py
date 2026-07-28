"""
hull_moving_average.py
======================
Tool 45 — Hull Moving Average (HMA)

A low-lag moving average that is extremely popular for high-frequency 
scalping due to its speed.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class HMATool(BaseTool):
    name = "hma"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M5") or buffers.get("M1")
        if df is None or len(df) < 20:
            return result

        close = df["close"]
        n = 14
        
        def wma(s, period):
            weights = np.arange(1, period + 1)
            return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
            
        hma = wma(2 * wma(close, n // 2) - wma(close, n), int(np.sqrt(n)))
        
        last_hma = hma.iloc[-1]
        prev_hma = hma.iloc[-2]
        
        # Scoring: +1 if HMA is pointing up, -1 if down
        slope = last_hma - prev_hma
        result.score = float(np.tanh(slope * 1000))
        result.features = {
            "hma_slope": result.score,
            "price_vs_hma": float(close.iloc[-1] - last_hma)
        }
        return result
