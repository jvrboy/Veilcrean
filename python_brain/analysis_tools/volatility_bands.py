"""
volatility_bands.py
===================
Tool 17 — Volatility Bands (Bollinger & Keltner)

Identifies overextension and volatility squeezes.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class VolatilityBandsTool(BaseTool):
    name = "volatility_bands"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M5") or buffers.get("M1")
        if df is None or len(df) < 30:
            return result

        close = df["close"]
        
        # Bollinger Bands
        ma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper_bb = ma + (std * 2)
        lower_bb = ma - (std * 2)
        
        # Keltner Channels (ATR based)
        high_low = df["high"] - df["low"]
        atr = high_low.rolling(20).mean()
        upper_kc = ma + (atr * 1.5)
        lower_kc = ma - (atr * 1.5)
        
        curr_p = ctx.get("price", close.iloc[-1])
        
        # Score Logic
        score = 0.0
        # Overbought (Outside BB and KC)
        if curr_p > upper_bb.iloc[-1] and curr_p > upper_kc.iloc[-1]:
            score = -0.9
        # Oversold
        elif curr_p < lower_bb.iloc[-1] and curr_p < lower_kc.iloc[-1]:
            score = 0.9
            
        # Squeeze detection: BB inside KC
        squeeze = std.iloc[-1] * 2 < atr.iloc[-1] * 1.5
        
        result.score = float(score)
        result.confidence = 0.75
        result.features = {
            "bb_kc_squeeze": float(squeeze),
            "bb_pos": float((curr_p - ma.iloc[-1]) / (std.iloc[-1] * 2 + 1e-9))
        }
        return result
