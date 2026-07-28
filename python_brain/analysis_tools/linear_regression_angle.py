"""
linear_regression_angle.py
==========================
Tool 146 — Linear Regression Angle

Calculates the angle of the linear regression slope in degrees to 
quantify trend steepness.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class LinRegAngleTool(BaseTool):
    name = "lin_reg_angle"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20: return res

        n = 14
        y = df["close"].tail(n).values
        x = np.arange(n)
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        
        # Angle in degrees
        angle = np.rad2deg(np.arctan(slope))
        
        res.score = float(np.tanh(angle / 45.0))
        res.features = {"lra_angle": float(angle)}
        return res
