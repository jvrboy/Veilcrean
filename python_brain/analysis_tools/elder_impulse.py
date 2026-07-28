"""
elder_impulse.py
================
Tool 116 — Elder Impulse System

Combines a 13-period EMA and the MACD-Histogram to color price bars.
Identifies when both trend and momentum are in agreement.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class ElderImpulseTool(BaseTool):
    name = "elder_impulse"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 30:
            return result

        close = df["close"]
        ema13 = close.ewm(span=13, adjust=False).mean()
        
        # MACD Histogram
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        
        # Impulse logic
        # Green: EMA(13) rising and MACD-Hist rising
        # Red: EMA(13) falling and MACD-Hist falling
        # Blue: Neutral (mixed)
        
        ema_rising = ema13.iloc[-1] > ema13.iloc[-2]
        hist_rising = hist.iloc[-1] > hist.iloc[-2]
        
        score = 0.0
        if ema_rising and hist_rising: score = 1.0
        elif not ema_rising and not hist_rising: score = -1.0
        
        result.score = score
        result.features = {
            "impulse_score": score,
            "ema_13_rising": float(ema_rising),
            "hist_rising": float(hist_rising)
        }
        return result
