"""
center_of_gravity.py
====================
Tool 65 — Center of Gravity (CoG) Oscillator

An oscillator created by John Ehlers that identifies turning points 
with zero lag.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class CoGTool(BaseTool):
    name = "cog"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 10:
            return result

        n = 10
        prices = df["close"].tail(n).values
        
        num = 0
        den = 0
        for i in range(n):
            num += (i + 1) * prices[-(i + 1)]
            den += prices[-(i + 1)]
            
        cog = -num / (den + 1e-9)
        
        result.score = float(np.tanh(cog))
        result.features = {"cog_val": float(cog)}
        return result
