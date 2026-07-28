"""
directional_trend_index.py
==========================
Tool 134 — Directional Trend Index (DTI)

William Blau's trend indicator that uses double-smoothed price differences 
to identify trend strength and direction.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class DTITool(BaseTool):
    name = "dti"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 30: return res

        high = df["high"]
        low = df["low"]
        
        # Up Move = high - high[1]
        # Down Move = low[1] - low
        up = (high - high.shift(1)).clip(lower=0)
        down = (low.shift(1) - low).clip(lower=0)
        
        # Double smoothing
        r, s = 10, 10
        num = (up - down).ewm(span=r).mean().ewm(span=s).mean()
        den = (up - down).abs().ewm(span=r).mean().ewm(span=s).mean()
        
        dti = 100 * (num / (den + 1e-9))
        last_dti = dti.iloc[-1]
        
        res.score = float(last_dti / 100.0)
        res.features = {"dti_val": float(last_dti)}
        return res
