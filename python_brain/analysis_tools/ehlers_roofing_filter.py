"""
ehlers_roofing_filter.py
========================
Tool 140 — Ehlers Roofing Filter

A specialized filter that acts as a 'Roof' over price data, removing both 
spectral noise and trend components to leave a pure cycle.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class RoofingFilterTool(BaseTool):
    name = "roofing_filter"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 50: return res

        # Roofing filter = HighPass -> SuperSmoother
        prices = (df["high"] + df["low"]) / 2
        alpha1 = 0.07
        
        hp = np.zeros_like(prices)
        for i in range(1, len(prices)):
            hp[i] = (1 - alpha1/2) * (prices.iloc[i] - prices.iloc[i-1]) + (1 - alpha1) * hp[i-1]
            
        # SuperSmoother on hp
        a1 = np.exp(-1.414 * 3.14 / 10)
        b1 = 2 * a1 * np.cos(1.414 * 3.14 / 10)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        filt = np.zeros_like(hp)
        for i in range(2, len(hp)):
            filt[i] = c1 * (hp[i] + hp[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
            
        last_filt = filt[-1]
        
        res.score = float(np.tanh(last_filt * 100))
        res.features = {"roof_val": float(last_filt)}
        return res
