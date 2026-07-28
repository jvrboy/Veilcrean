"""
stdev_projection.py
===================
Tool 24 — Standard Deviation Projection

Uses historical volatility (Standard Deviation) to project the likely
high and low boundaries for the current/next candle.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult

class StdevProjectionTool(BaseTool):
    name = "stdev_projection"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 50:
            return result

        # 1. Calculate historical SD of returns
        returns = df["close"].pct_change().dropna()
        volatility = returns.std()
        
        # 2. Project boundaries
        last_close = df["close"].iloc[-1]
        upper_1sd = last_close * (1 + volatility)
        lower_1sd = last_close * (1 - volatility)
        upper_2sd = last_close * (1 + volatility * 2)
        lower_2sd = last_close * (1 - volatility * 2)
        
        current_p = ctx.get("price", last_close)
        
        # Score: +1 if at 2SD Low, -1 if at 2SD High
        score = 0.0
        if current_p >= upper_2sd:
            score = -1.0 # Exhausted high
        elif current_p <= lower_2sd:
            score = 1.0  # Exhausted low
        elif current_p >= upper_1sd:
            score = -0.5
        elif current_p <= lower_1sd:
            score = 0.5

        result.score = float(score)
        result.confidence = 0.8
        result.features = {
            "dist_to_2sd_upper": float(upper_2sd - current_p),
            "dist_to_2sd_lower": float(current_p - lower_2sd),
            "volatility_index":  float(volatility * 1000)
        }
        result.metadata = {"upper_2sd": upper_2sd, "lower_2sd": lower_2sd}
        
        return result
