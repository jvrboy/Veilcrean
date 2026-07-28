"""
fractal_dimension_index.py
==========================
Tool 123 — Fractal Dimension Index (FDI)

Measures the 'Complexity' or 'Fractal Dimension' of price action. 
FDI close to 1.0 indicates a strong trend; FDI close to 2.0 indicates 
random, choppy noise.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class FDITool(BaseTool):
    name = "fdi"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 30:
            return result

        n = 30
        close = df["close"].tail(n).values
        
        # Calculate length
        diffs = np.abs(close[1:] - close[:-1])
        length = np.sum(diffs)
        
        # Range
        high = np.max(close)
        low = np.min(close)
        rng = high - low
        
        # Fractal Dimension Estimate
        if rng == 0: return result
        fdi = 1.0 + (np.log(length / rng) / np.log(n))
        
        result.score = 0.0 # Dimension measure
        result.features = {"fdi_val": float(fdi)}
        result.metadata = {"market_state": "TRENDING" if fdi < 1.5 else "RANDOM"}
        return result
