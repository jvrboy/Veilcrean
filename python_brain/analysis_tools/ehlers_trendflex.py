"""
ehlers_trendflex.py
===================
Tool 139 — Ehlers Trendflex

An advanced trend-following oscillator that is essentially a Zero-Lag 
SuperSmoother filter designed for early trend detection.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class TrendflexTool(BaseTool):
    name = "trendflex"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 50: return res

        close = df["close"]
        a1 = np.exp(-1.414 * 3.14159 / 10.0)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / 10.0)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        tf = np.zeros_like(close)
        for i in range(2, len(close)):
            tf[i] = c1 * (close.iloc[i] + close.iloc[i-1]) / 2 + c2 * tf[i-1] + c3 * tf[i-2]
            
        last_tf = tf[-1]
        prev_tf = tf[-2]
        
        res.score = float(np.tanh((last_tf - prev_tf) * 500))
        res.features = {"trendflex_val": float(last_tf)}
        return res
