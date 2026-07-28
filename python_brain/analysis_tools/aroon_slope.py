"""
aroon_slope.py
==============
Tool 106 — Aroon Slope

Measures the rate of change of the Aroon Oscillator to identify accelerating 
trend changes.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class AroonSlopeTool(BaseTool):
    name = "aroon_slope"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 30:
            return result

        n = 25
        up = df["high"].rolling(n + 1).apply(lambda x: float(np.argmax(x)) / n * 100, raw=True)
        down = df["low"].rolling(n + 1).apply(lambda x: float(np.argmin(x)) / n * 100, raw=True)
        
        osc = up - down
        slope = osc.diff(5) / 5.0
        
        last_slope = slope.iloc[-1]
        
        result.score = float(np.tanh(last_slope / 10.0))
        result.features = {"aroon_slope": float(last_slope)}
        return result
