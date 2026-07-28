"""
lin_reg_intercept.py
====================
Tool 86 — Linear Regression Intercept

Calculates the intercept of the linear regression line, helping the bot
determine the mathematical base price relative to current action.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class LinRegInterceptTool(BaseTool):
    name = "lin_reg_intercept"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        n = 20
        y = df["close"].tail(n).values
        x = np.arange(n)
        
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        
        price = df["close"].iloc[-1]
        
        result.score = float(np.tanh((price - intercept) / (price * 0.01 + 1e-9)))
        result.features = {
            "lri_intercept": float(intercept),
            "lri_diff": float(price - intercept)
        }
        return result
