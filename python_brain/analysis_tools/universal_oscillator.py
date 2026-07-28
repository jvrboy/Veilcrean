"""
universal_oscillator.py
========================
Tool 135 — Ehlers Universal Oscillator

A specialized John Ehlers oscillator that uses a Bandpass filter to identify 
market cycles across any symbol.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class UniversalOscillatorTool(BaseTool):
    name = "universal_osc"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 30: return res

        close = df["close"]
        
        # Simplified bandpass proxy: (EMA(fast) - EMA(slow)) / ATR
        ema_f = close.ewm(span=10).mean()
        ema_s = close.ewm(span=20).mean()
        
        tr = (df["high"] - df["low"]).rolling(14).mean()
        
        u_osc = (ema_f - ema_s) / (tr + 1e-9)
        
        last_u = u_osc.iloc[-1]
        
        res.score = float(np.tanh(last_u * 20))
        res.features = {"universal_osc_val": float(last_u)}
        return res
