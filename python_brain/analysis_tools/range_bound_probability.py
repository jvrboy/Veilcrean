"""
range_bound_probability.py
==========================
Tool 115 — Range-Bound Probability

Mathematically estimates the likelihood that price will remain within 
a specific range over the next N bars.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class RangeBoundTool(BaseTool):
    name = "range_bound"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 50:
            return result

        close = df["close"]
        
        # 1. Variance of returns
        returns = close.pct_change().dropna()
        sigma = returns.std()
        
        # 2. Probability of staying within 1 ATR
        tr = (df["high"] - df["low"]).rolling(14).mean()
        atr = tr.iloc[-1]
        
        # Probability based on normal distribution of price drift
        # This is a simplified proxy
        dist_to_upper = (df["high"].rolling(20).max() - close).iloc[-1]
        dist_to_lower = (close - df["low"].rolling(20).min()).iloc[-1]
        
        range_stability = 1.0 - (atr / (dist_to_upper + dist_to_lower + 1e-9))
        
        result.score = 0.0 # Context indicator
        result.features = {"range_prob": float(np.clip(range_stability, 0, 1))}
        return result
