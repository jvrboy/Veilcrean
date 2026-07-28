"""
ehlers_cyber_cycle.py
=====================
Tool 136 — Ehlers Cyber Cycle

An oscillator that adapts to the market's dominant cycle using 
pre-smoothing and a unique cycle-detection logic.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class CyberCycleTool(BaseTool):
    name = "cyber_cycle"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20: return res

        prices = (df["high"] + df["low"]) / 2
        alpha = 0.07
        
        # Pre-smoothing (Ehlers Smooth)
        smooth = (prices + 2 * prices.shift(1) + 2 * prices.shift(2) + prices.shift(3)) / 6
        
        cycle = np.zeros_like(smooth)
        for i in range(2, len(smooth)):
            cycle[i] = (1 - 0.5 * alpha)**2 * (smooth.iloc[i] - 2 * smooth.iloc[i-1] + smooth.iloc[i-2]) + \
                       2 * (1 - alpha) * cycle[i-1] - (1 - alpha)**2 * cycle[i-2]
        
        last_cycle = cycle[-1]
        prev_cycle = cycle[-2]
        
        res.score = float(np.tanh(last_cycle * 100))
        res.features = {"cycle_val": float(last_cycle), "cycle_slope": float(last_cycle - prev_cycle)}
        return res
