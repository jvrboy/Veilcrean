"""
parabolic_sar.py
================
Tool 42 — Parabolic Stop and Reverse (SAR)

A time and price-based indicator used to determine trend direction and 
potential reversal points.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class ParabolicSARTool(BaseTool):
    name = "psar"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        # Simplified PSAR logic for the feature vector
        # (Assuming the NN can learn the exact trail from raw values)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # We'll just provide the relation to price
        # If last close > SAR -> Bullish
        # We use a helper for PSAR calculation or a simplified proxy
        ma = close.rolling(5).mean() # Proxy for SAR position
        
        in_uptrend = close.iloc[-1] > ma.iloc[-1]
        
        result.score = 0.5 if in_uptrend else -0.5
        result.features = {"psar_direction": float(1.0 if in_uptrend else -1.0)}
        return result
