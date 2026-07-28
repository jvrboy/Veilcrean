"""
balance_of_power.py
===================
Tool 73 — Balance of Power (BOP)

Measures the strength of the bulls vs bears by assessing the ability 
of each to push price to extreme levels.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class BOPTool(BaseTool):
    name = "bop"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None: return result

        # Formula: (Close - Open) / (High - Low)
        bop = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
        # Smoothed
        bop_sma = bop.rolling(14).mean()
        
        last_bop = bop_sma.iloc[-1]
        
        result.score = float(np.tanh(last_bop * 2))
        result.features = {"bop_val": float(last_bop)}
        return result
