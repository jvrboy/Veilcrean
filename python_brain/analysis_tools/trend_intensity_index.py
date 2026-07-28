"""
trend_intensity_index.py
========================
Tool 104 — Trend Intensity Index (TII)

Measures the percentage of price movement that occurs above or below 
a moving average to quantify trend strength.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class TIITool(BaseTool):
    name = "tii"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 60:
            return result

        n = 30 # MA period
        m = 60 # Intensity window
        
        close = df["close"]
        sma = close.rolling(n).mean()
        
        dev = close - sma
        
        pos_dev = dev.clip(lower=0).rolling(m).sum()
        neg_dev = (-dev.clip(upper=0)).rolling(m).sum()
        
        tii = 100 * (pos_dev / (pos_dev + neg_dev + 1e-9))
        
        last_tii = tii.iloc[-1]
        
        # TII > 80 (strong bullish), TII < 20 (strong bearish)
        result.score = float((last_tii - 50) / 50)
        result.features = {"tii_val": float(last_tii)}
        return result
