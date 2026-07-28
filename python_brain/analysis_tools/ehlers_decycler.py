"""
ehlers_decycler.py
==================
Tool 137 — Ehlers Decycler

A high-performance filter that removes the cycle components of price 
to leave only the underlying trend component.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class DecyclerTool(BaseTool):
    name = "decycler"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20: return res

        prices = df["close"]
        alpha1 = 0.07 # Standard alpha
        
        # High pass filter
        hp = np.zeros_like(prices)
        for i in range(1, len(prices)):
            hp[i] = (1 - alpha1/2) * (prices.iloc[i] - prices.iloc[i-1]) + (1 - alpha1) * hp[i-1]
            
        decycler = prices - hp
        
        last_dec = decycler.iloc[-1]
        prev_dec = decycler.iloc[-2]
        
        res.score = float(np.tanh((last_dec - prev_dec) * 1000))
        res.features = {"decycler_val": float(last_dec)}
        return res
