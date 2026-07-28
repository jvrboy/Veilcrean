"""
ichimoku_cloud.py
=================
Tool 36 — Ichimoku Kinko Hyo (Cloud)

A comprehensive indicator that defines support/resistance, trend 
direction, and momentum.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class IchimokuCloudTool(BaseTool):
    name = "ichimoku"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M30")
        if df is None or len(df) < 52:
            return result

        highs = df["high"]
        lows = df["low"]
        close = df["close"]

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        period9_high = highs.rolling(window=9).max()
        period9_low = lows.rolling(window=9).min()
        tenkan = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        period26_high = highs.rolling(window=26).max()
        period26_low = lows.rolling(window=26).min()
        kijun = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2
        span_a = ((tenkan + kijun) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
        period52_high = highs.rolling(window=52).max()
        period52_low = lows.rolling(window=52).min()
        span_b = ((period52_high + period52_low) / 2).shift(26)

        last_close = close.iloc[-1]
        last_a = span_a.iloc[-1]
        last_b = span_b.iloc[-1]
        
        # Trend: Bullish if price > Cloud (Span A & B)
        score = 0.0
        if last_close > max(last_a, last_b):
            score = 0.8
        elif last_close < min(last_a, last_b):
            score = -0.8
            
        result.score = score
        result.features = {
            "ichimoku_score": score,
            "above_cloud": float(last_close > max(last_a, last_b)),
            "tenkan_kijun_cross": float(tenkan.iloc[-1] - kijun.iloc[-1])
        }
        return result
