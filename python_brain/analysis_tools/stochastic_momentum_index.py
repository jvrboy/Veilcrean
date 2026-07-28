"""
stochastic_momentum_index.py
============================
Tool 103 — Stochastic Momentum Index (SMI)

A more refined version of the stochastic oscillator that measures the 
distance of the current close relative to the center of the recent high/low range.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class SMITool(BaseTool):
    name = "smi"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 30:
            return result

        n = 13 # Period
        m1 = 25 # Smoothing 1
        m2 = 2 # Smoothing 2
        
        high_n = df["high"].rolling(n).max()
        low_n  = df["low"].rolling(n).min()
        center = (high_n + low_n) / 2
        
        diff = df["close"] - center
        
        # Double smoothed diff
        num = diff.ewm(span=m1).mean().ewm(span=m2).mean()
        # Double smoothed range
        den = (high_n - low_n).ewm(span=m1).mean().ewm(span=m2).mean() / 2
        
        smi = 100 * (num / (den + 1e-9))
        signal = smi.ewm(span=m2).mean()
        
        last_smi = smi.iloc[-1]
        last_sig = signal.iloc[-1]
        
        result.score = float(np.tanh((last_smi - last_sig) / 10.0))
        result.features = {"smi_val": float(last_smi)}
        return result
