"""
tema.py
=======
Tool 51 — Triple Exponential Moving Average (TEMA)

A fast-reacting moving average that reduces lag even further than 
DEMA or standard EMA.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class TEMATool(BaseTool):
    name = "tema"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 30:
            return result

        n = 14
        close = df["close"]
        
        ema1 = close.ewm(span=n, adjust=False).mean()
        ema2 = ema1.ewm(span=n, adjust=False).mean()
        ema3 = ema2.ewm(span=n, adjust=False).mean()
        
        tema = 3 * ema1 - 3 * ema2 + ema3
        
        last_tema = tema.iloc[-1]
        prev_tema = tema.iloc[-2]
        
        result.score = float(np.tanh((last_tema - prev_tema) * 1000))
        result.features = {
            "tema_val": float(last_tema),
            "tema_slope": result.score
        }
        return result
