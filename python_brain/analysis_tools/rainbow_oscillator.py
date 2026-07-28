"""
rainbow_oscillator.py
=====================
Tool 96 — Rainbow Oscillator

Uses recursive smoothing (multiple levels of SMA) to identify trend 
extremes and potential reversals.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class RainbowOscillatorTool(BaseTool):
    name = "rainbow_osc"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        close = df["close"]
        
        # Multiple levels of SMA
        ma1 = close.rolling(2).mean()
        ma2 = ma1.rolling(2).mean()
        ma3 = ma2.rolling(2).mean()
        ma4 = ma3.rolling(2).mean()
        ma5 = ma4.rolling(2).mean()
        ma6 = ma5.rolling(2).mean()
        ma7 = ma6.rolling(2).mean()
        ma8 = ma7.rolling(2).mean()
        ma9 = ma8.rolling(2).mean()
        ma10 = ma9.rolling(2).mean()
        
        rainbow_max = pd.concat([ma1, ma2, ma3, ma4, ma5, ma6, ma7, ma8, ma9, ma10], axis=1).max(axis=1)
        rainbow_min = pd.concat([ma1, ma2, ma3, ma4, ma5, ma6, ma7, ma8, ma9, ma10], axis=1).min(axis=1)
        
        osc = 100 * (close - (ma1 + ma2 + ma3 + ma4 + ma5 + ma6 + ma7 + ma8 + ma9 + ma10)/10) / (rainbow_max - rainbow_min + 1e-9)
        
        last_osc = osc.iloc[-1]
        
        result.score = float(np.tanh(last_osc / 50.0))
        result.features = {"rainbow_val": float(last_osc)}
        return result
