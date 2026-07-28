"""
kaufman_adaptive_ma.py
======================
Tool 77 — Kaufman's Adaptive Moving Average (KAMA)

A moving average that automatically adjusts its speed based on price 
volatility and noise level.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class KAMATool(BaseTool):
    name = "kama"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 10:
            return result

        close = df["close"].values
        n = 10
        
        # Efficiency Ratio (ER)
        change = np.abs(close[1:] - close[:-1])
        volatility = pd.Series(change).rolling(n).sum()
        direction = np.abs(pd.Series(close).diff(n))
        er = direction / (volatility + 1e-9)
        
        # Smoothing Constant (SC)
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc)**2
        
        kama = np.zeros_like(close)
        kama[n] = close[n]
        for i in range(n + 1, len(close)):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
            
        last_kama = kama[-1]
        prev_kama = kama[-2]
        
        result.score = float(np.tanh((last_kama - prev_kama) * 1000))
        result.features = {
            "kama_val": float(last_kama),
            "kama_slope": result.score
        }
        return result
