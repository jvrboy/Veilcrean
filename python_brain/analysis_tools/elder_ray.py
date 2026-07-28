"""
elder_ray.py
============
Tool 38 — Elder-Ray Index (Bull/Bear Power)

Measures the buying and selling pressure in the market by comparing 
High/Low to an Exponential Moving Average.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class ElderRayTool(BaseTool):
    name = "elder_ray"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 13:
            return result

        ema = df["close"].ewm(span=13, adjust=False).mean()
        bull_power = df["high"] - ema
        bear_power = df["low"] - ema
        
        last_bull = bull_power.iloc[-1]
        last_bear = bear_power.iloc[-1]
        
        score = 0.0
        if last_bull > 0 and last_bear < 0:
            score = float(np.tanh((last_bull + last_bear) * 1000))
        elif last_bull > 0:
            score = 0.5
        elif last_bear < 0:
            score = -0.5
            
        result.score = float(np.clip(score, -1, 1))
        result.features = {
            "bull_power": float(last_bull),
            "bear_power": float(last_bear)
        }
        return result
