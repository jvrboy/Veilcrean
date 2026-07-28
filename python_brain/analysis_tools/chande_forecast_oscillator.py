"""
chande_forecast_oscillator.py
==============================
Tool 80 — Chande Forecast Oscillator (CFO)

Calculates the percentage difference between the actual price and the 
linear regression forecast price.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class CFOTool(BaseTool):
    name = "cfo"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 14:
            return result

        n = 14
        y = df["close"].tail(n).values
        x = np.arange(n)
        
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        forecast = slope * n + intercept # Predicted price for next bar
        
        current_p = y[-1]
        cfo = 100 * (current_p - forecast) / current_p
        
        result.score = -float(np.tanh(cfo / 1.0)) # Reversal bias if price exceeds forecast
        result.features = {"cfo_val": float(cfo)}
        return result
