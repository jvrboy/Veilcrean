"""
instantaneous_trendline.py
==========================
Tool 120 — Ehler's Instantaneous Trendline

A dominant cycle-aware moving average that filters out cyclic noise to show 
the underlying institutional trendline.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class InstantaneousTrendTool(BaseTool):
    name = "itrend"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        # Simplified Ehler's ITrend proxy
        prices = (df["high"] + df["low"]) / 2
        alpha = 0.07
        itrend = np.zeros_like(prices)
        for i in range(2, len(prices)):
            itrend[i] = (alpha - alpha**2 / 4) * prices.iloc[i] + (alpha**2 / 2) * prices.iloc[i-1] - (alpha - 3 * alpha**2 / 4) * prices.iloc[i-2] + 2 * (1 - alpha) * itrend[i-1] - (1 - alpha)**2 * itrend[i-2]
            
        last_it = itrend[-1]
        prev_it = itrend[-2]
        
        result.score = float(np.tanh((last_it - prev_it) * 1000))
        result.features = {"itrend_val": float(last_it)}
        return result
