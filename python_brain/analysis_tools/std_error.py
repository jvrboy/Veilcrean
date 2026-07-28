"""
std_error.py
============
Tool 90 — Standard Error

Measures the dispersion of prices around a linear regression line.
Used to detect abnormal volatility.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class StdErrorTool(BaseTool):
    name = "std_error"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        n = 20
        y = df["close"].tail(n).values
        x = np.arange(n)
        
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        
        result.score = 0.0 # Volatility measure
        result.features = {"std_error_val": float(std_err)}
        result.metadata = {"error_ratio": float(std_err / df["close"].iloc[-1])}
        return result
