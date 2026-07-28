"""
know_sure_thing.py
==================
Tool 61 — Know Sure Thing (KST)

A momentum oscillator based on the smoothed rate-of-change of four 
different timeframes.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class KSTTool(BaseTool):
    name = "kst"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 30:
            return result

        close = df["close"]
        
        def rcma(s, r, m):
            roc = ((s - s.shift(r)) / s.shift(r)) * 100
            return roc.rolling(m).mean()
            
        k1 = rcma(close, 10, 10)
        k2 = rcma(close, 15, 10)
        k3 = rcma(close, 20, 10)
        k4 = rcma(close, 30, 15)
        
        kst = k1 + 2*k2 + 3*k3 + 4*k4
        signal = kst.rolling(9).mean()
        
        last_kst = kst.iloc[-1]
        last_sig = signal.iloc[-1]
        
        result.score = float(np.tanh((last_kst - last_sig) / 10.0))
        result.features = {"kst_val": float(last_kst)}
        return result
