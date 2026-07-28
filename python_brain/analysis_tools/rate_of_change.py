"""
rate_of_change.py
=================
Tool 50 — Rate of Change (ROC)

Measures the percentage change between the current price and the price 
n-periods ago.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class ROCTool(BaseTool):
    name = "roc"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 14:
            return result

        n = 14
        roc = ((df["close"] - df["close"].shift(n)) / df["close"].shift(n)) * 100
        
        roc_val = roc.iloc[-1]
        
        result.score = float(np.tanh(roc_val / 5.0))
        result.features = {"roc_val": float(roc_val)}
        return result
