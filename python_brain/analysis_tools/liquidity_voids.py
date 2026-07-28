"""
liquidity_voids.py
==================
Tool 20 — Liquidity Voids / Imbalances

Detects large, fast candles that leave "holes" in the market, which 
price tends to fill later.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class LiquidityVoidTool(BaseTool):
    name = "liquidity_void"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("M5")
        if df is None or len(df) < 10:
            return result

        # A void is a candle whose body is significantly larger than its neighbors
        bodies = np.abs(df["close"] - df["open"])
        avg_body = bodies.rolling(10).mean().iloc[-1]
        
        last_body = bodies.iloc[-1]
        
        score = 0.0
        if last_body > avg_body * 3:
            # If bullish void, expect price to eventually return (bearish score)
            if df["close"].iloc[-1] > df["open"].iloc[-1]:
                score = -0.3
            else:
                score = 0.3
                
        result.score = float(score)
        result.features = {"void_ratio": float(last_body / (avg_body + 1e-9))}
        return result
