"""
lin_reg_curve.py
================
Tool 121 — Linear Regression Curve (LRC)

Calculates the end value of a linear regression line for each bar, 
plotting a smooth curve that follows price with less lag than an SMA.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class LinRegCurveTool(BaseTool):
    name = "lin_reg_curve"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        n = 14
        close = df["close"].values
        lrc = np.zeros_like(close)
        
        # We calculate the curve point (slope * current_index + intercept)
        for i in range(n, len(close)):
            y = close[i-n:i]
            x = np.arange(n)
            slope, intercept, r_val, p_val, std_err = linregress(x, y)
            lrc[i] = slope * (n-1) + intercept
            
        last_lrc = lrc[-1]
        prev_lrc = lrc[-2]
        
        result.score = float(np.tanh((last_lrc - prev_lrc) / (close[-1] * 0.001 + 1e-9)))
        result.features = {"lrc_val": float(last_lrc)}
        return result
