"""
lin_reg_channels.py
===================
Tool 52 — Linear Regression Channels

Calculates a linear regression line and two parallel lines (upper/lower) 
based on standard deviation to identify trend channels.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class LinRegChannelTool(BaseTool):
    name = "lin_reg_channel"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 50:
            return result

        n = 50
        y = df["close"].tail(n).values
        x = np.arange(n)
        
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        
        # Regression line values
        line = slope * x + intercept
        # Residuals to find SD
        residuals = y - line
        sd = np.std(residuals)
        
        upper = line[-1] + (2 * sd)
        lower = line[-1] - (2 * sd)
        
        price = df["close"].iloc[-1]
        
        # Scoring: +1 if at lower channel, -1 if at upper channel
        score = 0.0
        if price >= upper: score = -1.0
        elif price <= lower: score = 1.0
        else: score = -float((price - line[-1]) / (2 * sd + 1e-9))
            
        result.score = float(np.clip(score, -1, 1))
        result.features = {
            "lrc_pos": score,
            "lrc_slope": float(slope / price * 1000)
        }
        return result
