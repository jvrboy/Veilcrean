"""
micro_trend_regression.py
=========================
Tool 18 — Linear Regression Micro-Trend

Uses the slope of linear regression over a small window to detect 
high-frequency momentum shifts.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class MicroTrendTool(BaseTool):
    name = "micro_trend"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M1")
        if df is None or len(df) < 15:
            return result

        y = df["close"].tail(10).values
        x = np.arange(len(y))
        
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Normalize slope by price to get % change per bar
        norm_slope = slope / y[0] * 1000
        
        result.score = float(np.tanh(norm_slope * 5))
        result.confidence = float(abs(r_value)) # Use R-squared as confidence
        result.features = {
            "micro_slope": float(norm_slope),
            "micro_r2": float(r_value**2)
        }
        return result
