"""
donchian_channels.py
====================
Tool 33 — Donchian Channels

Calculates the highest high and lowest low over a period to identify
breakouts and range boundaries.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class DonchianChannelsTool(BaseTool):
    name = "donchian_channels"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        period = 20
        upper = df["high"].rolling(period).max()
        lower = df["low"].rolling(period).min()
        mid   = (upper + lower) / 2
        
        last_close = df["close"].iloc[-1]
        
        # Scoring: +1 if breakout above, -1 if breakout below
        score = 0.0
        if last_close >= upper.iloc[-1]:
            score = 1.0
        elif last_close <= lower.iloc[-1]:
            score = -1.0
        else:
            score = float((last_close - mid.iloc[-1]) / ((upper.iloc[-1] - lower.iloc[-1]) / 2 + 1e-9))
            
        result.score = float(np.clip(score, -1, 1))
        result.features = {
            "donchian_pos": score,
            "donchian_width": float((upper.iloc[-1] - lower.iloc[-1]) / last_close)
        }
        return result
