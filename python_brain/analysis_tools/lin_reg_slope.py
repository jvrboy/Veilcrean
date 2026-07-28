"""
lin_reg_slope.py
================
Tool 74 — Linear Regression Slope

Calculates the rate of change of the linear regression line to 
quantify momentum strength.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class LinRegSlopeTool(BaseTool):
    name = "lin_reg_slope"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        n = 20
        y = df["close"].tail(n).values
        x = np.arange(n)
        
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        
        result.score = float(np.tanh(slope / df["close"].iloc[-1] * 500))
        result.features = {
            "lrs_slope": float(slope),
            "lrs_r2": float(r_val**2)
        }
        return result
