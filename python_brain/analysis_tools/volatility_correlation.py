"""
volatility_correlation.py
=========================
Tool 27 — Volatility Correlation (VIX Proxy)

Analyzes the correlation between current price and overall market volatility.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult

class VolatilityCorrelationTool(BaseTool):
    name = "vol_correlation"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        # We compute a local "Volatility Index" from ATR
        # and check if price is moving with or against increasing volatility.
        high_low = df["high"] - df["low"]
        vol_index = high_low.rolling(14).mean()
        
        price_change = df["close"].diff()
        vol_change = vol_index.diff()
        
        # Check correlation over last 20 bars
        correlation = price_change.tail(20).corr(vol_change.tail(20))
        
        result.score = 0.0 # Neutral, providing features
        result.confidence = 1.0
        result.features = {"vol_price_correlation": float(np.nan_to_num(correlation, nan=0.0))}
        
        return result
