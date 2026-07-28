"""
pivot_points.py
===============
Tool 46 — Pivot Points (Classic)

Calculates the central pivot and its associated support (S1-S3) and 
resistance (R1-R3) levels for the current session.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class PivotPointsTool(BaseTool):
    name = "pivot_points"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("D1")
        if df is None or len(df) < 2:
            return result

        prev_day = df.iloc[-2]
        high = prev_day["high"]
        low = prev_day["low"]
        close = prev_day["close"]

        # Classic Pivot Point (P)
        p = (high + low + close) / 3
        
        r1 = (2 * p) - low
        s1 = (2 * p) - high
        
        r2 = p + (high - low)
        s2 = p - (high - low)
        
        r3 = high + 2 * (p - low)
        s3 = low - 2 * (high - p)
        
        price = ctx.get("price", close)
        
        # Scoring: +1 if at S3, -1 if at R3
        score = 0.0
        if price >= r3: score = -1.0
        elif price <= s3: score = 1.0
        elif price >= r1: score = -0.3
        elif price <= s1: score = 0.3
            
        result.score = score
        result.features = {
            "dist_to_p": float(price - p),
            "pivot_val": float(p)
        }
        result.metadata = {"levels": {"P": p, "R1": r1, "S1": s1}}
        return result
