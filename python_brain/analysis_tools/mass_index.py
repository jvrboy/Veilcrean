"""
mass_index.py
=============
Tool 63 — Mass Index

Used to predict trend reversals by analyzing the range between high 
and low prices.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class MassIndexTool(BaseTool):
    name = "mass_index"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 30:
            return result

        high_low = df["high"] - df["low"]
        ema9 = high_low.ewm(span=9, adjust=False).mean()
        ema9_9 = ema9.ewm(span=9, adjust=False).mean()
        
        ratio = ema9 / ema9_9
        mass_idx = ratio.rolling(25).sum()
        
        last_mi = mass_idx.iloc[-1]
        
        # A reversal is expected when Mass Index rises above 27 and then falls below 26.5
        result.score = 0.0
        result.confidence = 1.0
        result.features = {"mass_index": float(last_mi)}
        return result
