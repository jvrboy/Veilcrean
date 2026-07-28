"""
lin_reg_r2.py
=============
Tool 78 — Linear Regression R-Squared (R2)

Measures the reliability of the current trend. A high R2 indicates a strong, 
stable trend, while a low R2 suggests noise or ranging.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class LinRegR2Tool(BaseTool):
    name = "lin_reg_r2"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        n = 20
        y = df["close"].tail(n).values
        x = np.arange(n)
        
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        r2 = r_val**2
        
        result.score = 0.0 # Confidence indicator
        result.features = {"r2_val": float(r2)}
        result.metadata = {"trend_reliability": float(r2)}
        return result
