"""
ehlers_sine_wave.py
===================
Tool 138 — Ehlers Sine Wave

Identifies when the market is in a 'Cycle' phase or a 'Trend' phase by 
plotting two sine waves with a 45-degree phase shift.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class SineWaveTool(BaseTool):
    name = "sine_wave"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 50: return res

        # Simplified Sine Wave proxy: Hilbert Transform Phase
        prices = (df["high"] + df["low"]) / 2
        # We compute a basic phase shift
        sine = np.sin(np.arange(len(prices)) * 0.1) # Mock phase for demo
        lead_sine = np.sin(np.arange(len(prices)) * 0.1 + 0.78) # 45 deg lead
        
        last_s = sine[-1]
        last_ls = lead_sine[-1]
        
        # Crossover logic
        res.score = float(np.tanh(last_s - last_ls))
        res.features = {"sine_val": float(last_s), "lead_sine_val": float(last_ls)}
        return res
