"""
detrended_price_oscillator.py
=============================
Tool 58 — Detrended Price Oscillator (DPO)

An oscillator that removes trend from price to make it easier to identify 
cycles and overbought/oversold levels.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class DPOTool(BaseTool):
    name = "dpo"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 21:
            return result

        n = 20
        # DPO = Close - SMA(n / 2 + 1) backshifted (n / 2 + 1)
        shift = int(n / 2) + 1
        sma = df["close"].rolling(n).mean().shift(shift)
        dpo = df["close"] - sma
        
        last_dpo = dpo.iloc[-1]
        
        result.score = float(np.tanh(last_dpo / (df["close"].std() + 1e-9)))
        result.features = {"dpo_val": float(last_dpo)}
        return result
