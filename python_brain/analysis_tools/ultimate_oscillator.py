"""
ultimate_oscillator.py
======================
Tool 66 — Ultimate Oscillator

A momentum oscillator that uses three different timeframes to reduce 
the false signals common in other oscillators.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class UltimateOscillatorTool(BaseTool):
    name = "ultimate_osc"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 30:
            return result

        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        
        tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)
        bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
        
        def avg(period):
            return bp.rolling(period).sum() / tr.rolling(period).sum()
            
        uo = 100 * (4 * avg(7) + 2 * avg(14) + avg(28)) / (4 + 2 + 1)
        
        last_uo = uo.iloc[-1]
        
        result.score = float((last_uo - 50) / 50)
        result.features = {"uo_val": float(last_uo)}
        return result
