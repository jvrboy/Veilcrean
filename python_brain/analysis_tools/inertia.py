"""
inertia.py
==========
Tool 89 — Inertia Indicator

Calculates the linear regression of the Relative Vigor Index (RVI) 
to measure trend inertia.
"""
from __future__ import annotations
from typing import Dict
from scipy.stats import linregress
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class InertiaTool(BaseTool):
    name = "inertia"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 30:
            return result

        # 1. RVI
        close_open = df["close"] - df["open"]
        high_low = df["high"] - df["low"]
        rvi = close_open.rolling(10).mean() / (high_low.rolling(10).mean() + 1e-9)
        
        # 2. Inertia (LinReg of RVI)
        n = 20
        y = rvi.tail(n).values
        x = np.arange(n)
        slope, intercept, r_val, p_val, std_err = linregress(x, y)
        
        inertia_val = slope * n + intercept
        
        result.score = float(np.tanh(inertia_val * 10))
        result.features = {"inertia_val": float(inertia_val)}
        return result
