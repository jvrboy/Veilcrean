"""
squeeze_momentum.py
===================
Tool 113 — Squeeze Momentum (LazyBear style)

A popular tool that identifies volatility squeezes (Bollinger inside Keltner) 
and displays momentum as a colored histogram.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class SqueezeMomentumTool(BaseTool):
    name = "squeeze_momentum"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        period = 20
        mult = 2.0
        atr_mult = 1.5
        
        close = df["close"]
        ma = close.rolling(period).mean()
        
        # Bollinger
        std = close.rolling(period).std()
        upper_bb = ma + (mult * std)
        lower_bb = ma - (mult * std)
        
        # Keltner
        tr = pd.concat([df["high"] - df["low"], (df["high"] - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        upper_kc = ma + (atr_mult * atr)
        lower_kc = ma - (atr_mult * atr)
        
        # Momentum: LinReg of price from median
        highest = df["high"].rolling(period).max()
        lowest = df["low"].rolling(period).min()
        median = (highest + lowest) / 2
        avg = (median + ma) / 2
        mom = close - avg
        
        # Squeeze logic
        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        
        last_mom = mom.iloc[-1]
        
        result.score = float(np.tanh(last_mom / (atr.iloc[-1] + 1e-9)))
        result.features = {
            "sqz_mom": result.score,
            "sqz_on": float(sqz_on.iloc[-1])
        }
        return result
