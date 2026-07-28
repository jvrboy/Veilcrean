"""
ehlers_cti.py
=============
Tool 150 — Ehlers Correlation Trend Indicator (CTI)

A trend indicator that uses the Spearman correlation between price 
and a straight-line trend to identify trend direction and quality.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class CTITool(BaseTool):
    name = "cti"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20: return res

        n = 20
        close = df["close"].tail(n).values
        # Straight line trend [0, 1, 2, ..., n-1]
        trend = np.arange(n)
        
        # Calculate correlation
        if len(close) == n:
            corr = np.corrcoef(close, trend)[0, 1]
            res.score = float(corr)
            res.features = {"cti_val": float(corr)}
        
        return res
