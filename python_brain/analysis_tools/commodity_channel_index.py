"""
commodity_channel_index.py
==========================
Tool 72 — Commodity Channel Index (CCI)

Measures the current price level relative to an average price level 
over a given period of time.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class CCITool(BaseTool):
    name = "cci"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        n = 20
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(n).mean()
        # Mean Deviation
        mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        cci = (tp - sma_tp) / (0.015 * mad + 1e-9)
        
        cci_val = cci.iloc[-1]
        
        result.score = float(np.tanh(cci_val / 100.0))
        result.features = {"cci_val": float(cci_val)}
        return result
