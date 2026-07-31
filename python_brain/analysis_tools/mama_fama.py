"""
mama_fama.py
============
Tool 101 & 102 — MESA Adaptive Moving Average (MAMA) & Following Adaptive Moving Average (FAMA)

John Ehlers' adaptive moving average that uses the Hilbert Transform to distinguish 
between market cycles and trend. MAMA adapts to price changes very quickly, 
while FAMA follows MAMA with a slight lag, creating a crossover signal.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class MAMAFAMATool(BaseTool):
    name = "mama_fama"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 50:
            return result

        # Simplified MAMA/FAMA implementation
        # Uses Fast Limit and Slow Limit to control adaptation
        fast_limit = 0.5
        slow_limit = 0.05
        
        prices = (df["high"] + df["low"]) / 2
        mama = np.zeros_like(prices)
        fama = np.zeros_like(prices)
        
        # In a real implementation we'd use the Hilbert Transform Phase 
        # For the feature vector, we simulate adaptation via Efficiency Ratio
        change = prices.diff().abs()
        volatility = change.rolling(10).sum()
        er = (prices.diff(10).abs() / (volatility + 1e-9)).clip(0, 1)
        # rolling() leaves leading NaNs that would poison the recursive MAMA/FAMA.
        er = er.fillna(0.0)
        
        # Adaptive alpha
        alpha = er * (fast_limit - slow_limit) + slow_limit
        
        for i in range(1, len(prices)):
            mama[i] = alpha.iloc[i] * prices.iloc[i] + (1 - alpha.iloc[i]) * mama[i-1]
            fama[i] = 0.5 * alpha.iloc[i] * mama[i] + (1 - 0.5 * alpha.iloc[i]) * fama[i-1]
            
        last_mama = mama[-1]
        last_fama = fama[-1]
        
        result.score = float(np.tanh((last_mama - last_fama) / (prices.std() + 1e-9) * 10))
        result.features = {
            "mama_val": float(last_mama),
            "fama_val": float(last_fama),
            "mama_fama_diff": float(last_mama - last_fama)
        }
        return result
