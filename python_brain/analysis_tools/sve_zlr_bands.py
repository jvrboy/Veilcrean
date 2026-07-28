"""
sve_zlr_bands.py
================
Tool 133 — SVE_ZLR Percent Bands

Sylvain Vervoort's Zero-Lag Percent Bands designed for high-frequency 
trading to identify breakouts with minimal delay.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class SVEZLRBandsTool(BaseTool):
    name = "sve_zlr_bands"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20: return res

        n = 14
        close = df["close"]
        
        # Zero-lag proxy
        ema1 = close.ewm(span=n, adjust=False).mean()
        ema2 = ema1.ewm(span=n, adjust=False).mean()
        zlr = ema1 + (ema1 - ema2)
        
        # Bands
        atr = (df["high"] - df["low"]).rolling(n).mean()
        upper = zlr + (atr * 2.0)
        lower = zlr - (atr * 2.0)
        
        last_c = close.iloc[-1]
        last_u = upper.iloc[-1]
        last_l = lower.iloc[-1]
        
        res.score = float((last_c - zlr.iloc[-1]) / (atr.iloc[-1] * 2 + 1e-9))
        res.features = {
            "zlr_pos": res.score,
            "zlr_break": float(1.0 if last_c > last_u else -1.0 if last_c < last_l else 0.0)
        }
        return res
