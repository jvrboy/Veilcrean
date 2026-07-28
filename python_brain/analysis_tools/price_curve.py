"""
price_curve.py
==============
Tool 108 — Price Curve Acceleration

Fits a polynomial curve to recent prices to detect whether price 
acceleration is increasing or decreasing.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class PriceCurveTool(BaseTool):
    name = "price_curve"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        n = 15
        y = df["close"].tail(n).values
        x = np.arange(n)
        
        # Fit 2nd degree polynomial: y = ax^2 + bx + c
        poly = np.polyfit(x, y, 2)
        acceleration = 2 * poly[0] # The 'a' term represents acceleration
        
        result.score = float(np.tanh(acceleration / (y[-1] * 0.0001 + 1e-9)))
        result.features = {"price_accel": float(acceleration)}
        return result
