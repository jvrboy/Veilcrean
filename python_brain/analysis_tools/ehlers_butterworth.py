"""
ehlers_butterworth.py
=====================
Tool 142 — Ehlers 2nd Order Butterworth Filter

A classic low-pass filter that provides smooth trendlines with better 
noise suppression than standard moving averages.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class ButterworthTool(BaseTool):
    name = "butterworth"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20: return res

        close = df["close"]
        period = 15
        a1 = np.exp(-1.414 * 3.14159 / period)
        b1 = 2 * a1 * np.cos(1.414 * 3.14159 / period)
        c2 = b1
        c3 = -a1 * a1
        c1 = (1 - c2 - c3) / 4
        
        filt = np.zeros_like(close)
        for i in range(2, len(close)):
            filt[i] = c1 * (close.iloc[i] + 2 * close.iloc[i-1] + close.iloc[i-2]) + c2 * filt[i-1] + c3 * filt[i-2]
            
        last_f = filt[-1]
        prev_f = filt[-2]
        
        res.score = float(np.tanh((last_f - prev_f) * 1000))
        res.features = {"butter_val": float(last_f)}
        return res
