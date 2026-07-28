"""
williams_r.py
==============
Tool 44 — Williams %R

A momentum indicator that measures overbought and oversold levels.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class WilliamsRTool(BaseTool):
    name = "williams_r"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("M5")
        if df is None or len(df) < 14:
            return result

        n = 14
        hh = df["high"].rolling(n).max()
        ll = df["low"].rolling(n).min()
        close = df["close"]
        
        wr = -100 * (hh - close) / (hh - ll + 1e-9)
        
        wr_val = wr.iloc[-1]
        
        # Scoring: +1 if oversold (<-80), -1 if overbought (>-20)
        score = 0.0
        if wr_val < -80: score = 0.7
        elif wr_val > -20: score = -0.7
        
        result.score = score
        result.features = {"williams_r": float(wr_val)}
        return result
