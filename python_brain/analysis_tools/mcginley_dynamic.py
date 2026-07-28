"""
mcginley_dynamic.py
===================
Tool 56 — McGinley Dynamic

An advanced moving average designed to track the market better than standard
moving averages by adjusting for market speed (volatility).
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class McGinleyDynamicTool(BaseTool):
    name = "mcginley_dynamic"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        n = 14
        close = df["close"].values
        md = np.zeros_like(close)
        md[0] = close[0]
        
        # Formula: MD[i] = MD[i-1] + (Price[i] - MD[i-1]) / (N * (Price[i]/MD[i-1])^4)
        for i in range(1, len(close)):
            k = n * (close[i] / md[i-1])**4
            md[i] = md[i-1] + (close[i] - md[i-1]) / k
            
        last_md = md[-1]
        prev_md = md[-2]
        
        result.score = float(np.tanh((last_md - prev_md) * 1000))
        result.features = {
            "mcginley_val": float(last_md),
            "mcginley_slope": result.score
        }
        return result
