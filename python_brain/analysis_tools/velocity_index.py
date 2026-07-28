"""
velocity_index.py
=================
Tool 124 — Momentum Velocity Index

Measures the distance price travels per unit of time, normalized by ATR,
identifying the "Velocity" of market participants.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class VelocityTool(BaseTool):
    name = "velocity"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 14:
            return result

        n = 14
        close = df["close"]
        high_low = df["high"] - df["low"]
        atr = high_low.rolling(n).mean()
        
        velocity = (close - close.shift(n)) / (atr * n + 1e-9)
        
        last_v = velocity.iloc[-1]
        
        result.score = float(np.tanh(last_v * 10))
        result.features = {"velocity_val": float(last_v)}
        return result
