"""
true_strength_index.py
======================
Tool 60 — True Strength Index (TSI)

A momentum oscillator based on a double-smoothed moving average of 
price changes.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class TSITool(BaseTool):
    name = "tsi"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 30:
            return result

        diff = df["close"].diff()
        
        # Double smoothing for momentum
        ema1 = diff.ewm(span=25).mean()
        ema2 = ema1.ewm(span=13).mean()
        
        # Double smoothing for absolute momentum
        abs_ema1 = diff.abs().ewm(span=25).mean()
        abs_ema2 = abs_ema1.ewm(span=13).mean()
        
        tsi = 100 * (ema2 / (abs_ema2 + 1e-9))
        
        last_tsi = tsi.iloc[-1]
        
        result.score = float(last_tsi / 100.0)
        result.features = {"tsi_val": float(last_tsi)}
        return result
