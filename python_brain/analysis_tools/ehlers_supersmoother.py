"""
ehlers_supersmoother.py
=======================
Tool 141 — Ehlers SuperSmoother Filter

A superior alternative to standard moving averages. It uses a low-pass 
filter design to remove noise while preserving the trend with minimal lag.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class SuperSmootherTool(BaseTool):
    name = "supersmoother"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20: return res

        close = df["close"]
        a1 = np.exp(-1.414 * 3.14159 / 10.0)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / 10.0)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        filt = np.zeros_like(close)
        for i in range(2, len(close)):
            filt[i] = c1 * (close.iloc[i] + close.iloc[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
            
        last_f = filt[-1]
        prev_f = filt[-2]
        
        res.score = float(np.tanh((last_f - prev_f) * 1000))
        res.features = {"ss_val": float(last_f)}
        return res
