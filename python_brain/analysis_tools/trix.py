"""
trix.py
=======
Tool 64 — Trix (Triple Exponential Average)

Shows the rate-of-change of a triple-smoothed exponential moving average. 
Used to filter out minor price movements.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class TrixTool(BaseTool):
    name = "trix"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 45:
            return result

        n = 15
        close = df["close"]
        
        ema1 = close.ewm(span=n, adjust=False).mean()
        ema2 = ema1.ewm(span=n, adjust=False).mean()
        ema3 = ema2.ewm(span=n, adjust=False).mean()
        
        trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
        signal = trix.rolling(9).mean()
        
        last_trix = trix.iloc[-1]
        last_sig = signal.iloc[-1]
        
        result.score = float(np.tanh(last_trix - last_sig))
        result.features = {"trix_val": float(last_trix)}
        return result
