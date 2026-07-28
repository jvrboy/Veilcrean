"""
range_expansion_index.py
========================
Tool 122 — DeMark Range Expansion Index (REI)

An oscillator created by Thomas DeMark that identifies price exhaustion 
by comparing price moves across different time intervals.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class REITool(BaseTool):
    name = "rei"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 10:
            return result

        high = df["high"]
        low = df["low"]
        
        # DeMark REI logic
        num = (high - high.shift(2)) + (low - low.shift(2))
        den = (high - high.shift(2)).abs() + (low - low.shift(2)).abs()
        
        # Filters
        cond1 = (high >= low.shift(5)) | (high >= low.shift(6))
        cond2 = (low <= high.shift(5)) | (low <= high.shift(6))
        
        rei_num = np.where(cond1 & cond2, num, 0)
        
        rei = 100 * pd.Series(rei_num).rolling(8).sum() / pd.Series(den).rolling(8).sum()
        
        last_rei = rei.iloc[-1]
        
        # Scoring: Reversal bias above 60 or below -60
        result.score = -float(np.tanh(last_rei / 60.0))
        result.features = {"rei_val": float(last_rei)}
        return result
