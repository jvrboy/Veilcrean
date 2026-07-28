"""
lin_reg_forecast.py
===================
Tool 105 — Linear Regression Forecast (LRF)

Extends the linear regression line for one bar into the future to predict 
the next likely price level.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class LRFTool(BaseTool):
    name = "lin_reg_forecast"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        n = 14
        y = df["close"].tail(n).values
        x = np.arange(n)
        
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        
        # Forecast for next bar
        forecast = slope * n + intercept
        current = y[-1]
        
        result.score = float(np.tanh((forecast - current) / (current * 0.001 + 1e-9)))
        result.features = {
            "lrf_val": float(forecast),
            "lrf_diff_pct": float((forecast - current) / current)
        }
        return result
