"""
polarized_fractal_efficiency.py
===============================
Tool 131 — Polarized Fractal Efficiency (PFE)

Measures the efficiency of price movement. Values above 0 indicate 
an efficient bullish trend, while values below 0 indicate an efficient 
bearish trend.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class PFETool(BaseTool):
    name = "pfe"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20: return res

        n = 10
        m = 5
        close = df["close"]
        
        # Calculate PFE
        # Efficiency = Distance between ends / sum of step distances
        dx = (close - close.shift(n))
        dist = np.sqrt(dx**2 + n**2)
        
        # Sum of step distances
        steps = np.sqrt((close - close.shift(1))**2 + 1)
        sum_steps = steps.rolling(n).sum()
        
        pfe_raw = 100 * (dist / (sum_steps + 1e-9))
        pfe = pfe_raw * np.sign(dx)
        
        # Smooth
        pfe_smooth = pfe.rolling(m).mean()
        last_pfe = pfe_smooth.iloc[-1]
        
        res.score = float(last_pfe / 100.0)
        res.features = {"pfe_val": float(last_pfe)}
        return res
