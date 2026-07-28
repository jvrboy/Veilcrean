"""
efficiency_ratio.py
===================
Tool 93 — Efficiency Ratio (ER)

A stand-alone tool that quantifies trend "efficiency" by comparing net 
price movement to the total absolute movement.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class EfficiencyRatioTool(BaseTool):
    name = "efficiency_ratio"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 14:
            return result

        n = 14
        close = df["close"]
        net_change = (close - close.shift(n)).abs()
        total_volatility = (close.diff().abs()).rolling(n).sum()
        
        er = net_change / (total_volatility + 1e-9)
        
        last_er = er.iloc[-1]
        
        result.score = 0.0 # Metric for selecting strategy
        result.features = {"er_val": float(last_er)}
        result.metadata = {"efficiency": "HIGH" if last_er > 0.6 else "LOW"}
        return result
