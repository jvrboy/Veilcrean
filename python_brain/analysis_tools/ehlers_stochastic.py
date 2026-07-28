"""
ehlers_stochastic.py
====================
Tool 143 — Ehlers Stochastic (Cyber Cycle based)

Applies the Stochastic oscillator formula to the Cyber Cycle instead of 
raw price, providing a leading cyclic indicator.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class EhlersStochTool(BaseTool):
    name = "ehlers_stoch"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 50: return res

        prices = (df["high"] + df["low"]) / 2
        alpha = 0.07
        smooth = (prices + 2 * prices.shift(1) + 2 * prices.shift(2) + prices.shift(3)) / 6
        
        cycle = np.zeros_like(smooth)
        for i in range(2, len(smooth)):
            cycle[i] = (1 - 0.5 * alpha)**2 * (smooth.iloc[i] - 2 * smooth.iloc[i-1] + smooth.iloc[i-2]) + \
                       2 * (1 - alpha) * cycle[i-1] - (1 - alpha)**2 * cycle[i-2]
        
        cycle_s = pd.Series(cycle)
        n = 10
        stoch = (cycle_s - cycle_s.rolling(n).min()) / (cycle_s.rolling(n).max() - cycle_s.rolling(n).min() + 1e-9)
        
        last_s = stoch.iloc[-1]
        
        res.score = float((last_s - 0.5) * 2)
        res.features = {"ehlers_stoch_val": float(last_s)}
        return res
