"""
vidya.py
========
Tool 98 — Variable Index Dynamic Average (VIDYA)

A moving average that uses the Chande Momentum Oscillator (CMO) to adjust 
its smoothing factor.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class VIDYATool(BaseTool):
    name = "vidya"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        n = 9 # Period
        close = df["close"].values
        
        # 1. CMO
        diff = df["close"].diff()
        su = diff.clip(lower=0).rolling(n).sum()
        sd = (-diff.clip(upper=0)).rolling(n).sum()
        cmo = (su - sd).abs() / (su + sd + 1e-9)
        
        alpha = 2 / (n + 1)
        vidya = np.zeros_like(close)
        vidya[n] = close[n]
        
        for i in range(n + 1, len(close)):
            k = alpha * cmo.iloc[i]
            vidya[i] = k * close[i] + (1 - k) * vidya[i-1]
            
        last_vidya = vidya[-1]
        prev_vidya = vidya[-2]
        
        result.score = float(np.tanh((last_vidya - prev_vidya) * 1000))
        result.features = {
            "vidya_val": float(last_vidya),
            "vidya_slope": result.score
        }
        return result
