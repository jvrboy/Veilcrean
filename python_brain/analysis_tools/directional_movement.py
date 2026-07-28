"""
directional_movement.py
========================
Tool 107 — Directional Movement (DMI)

Focuses on the +DI and -DI components of the DMI system to identify 
directional dominance.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class DMITool(BaseTool):
    name = "dmi"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        n = 14
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm.loc[plus_dm < minus_dm] = 0
        minus_dm.loc[minus_dm < plus_dm] = 0
        
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr_n = tr.rolling(n).mean()
        
        plus_di = 100 * (plus_dm.rolling(n).mean() / (atr_n + 1e-9))
        minus_di = 100 * (minus_dm.rolling(n).mean() / (atr_n + 1e-9))
        
        last_p = plus_di.iloc[-1]
        last_m = minus_di.iloc[-1]
        
        result.score = float((last_p - last_m) / (last_p + last_m + 1e-9))
        result.features = {
            "plus_di": float(last_p),
            "minus_di": float(last_m)
        }
        return result
