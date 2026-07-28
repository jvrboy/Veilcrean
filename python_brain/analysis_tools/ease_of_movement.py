"""
ease_of_movement.py
===================
Tool 67 — Ease of Movement (EMV)

Relates price change to volume, identifying how "easily" the price 
is moving in a particular direction.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class EMVTool(BaseTool):
    name = "emv"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 14:
            return result

        high = df["high"]
        low = df["low"]
        vol = df["volume"] if "volume" in df.columns else (high - low)
        
        dm = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        br = (vol / 1000000) / (high - low + 1e-9)
        emv = dm / (br + 1e-9)
        emv_sma = emv.rolling(14).mean()
        
        last_emv = emv_sma.iloc[-1]
        
        result.score = float(np.tanh(last_emv))
        result.features = {"emv_val": float(last_emv)}
        return result
