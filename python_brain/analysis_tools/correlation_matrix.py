"""
correlation_matrix.py
=====================
Tool 16 — Correlation Matrix

Compares current symbol with multiple majors to find the strongest/weakest 
alignment.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class CorrelationMatrixTool(BaseTool):
    name = "correlation_matrix"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        symbol = ctx.get("symbol", "")
        
        majors = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]
        
        scores = {}
        for m in majors:
            df = buffers.get(f"{m}_H1")
            if df is not None and len(df) > 10:
                # Simple trend score for each major
                trend = 1.0 if df["close"].iloc[-1] > df["close"].iloc[-10] else -1.0
                scores[m] = trend
        
        if not scores: return result
        
        # Calculate "Global USD Strength"
        usd_strength = 0.0
        for m, trend in scores.items():
            if m.startswith("USD"): usd_strength += trend
            else: usd_strength -= trend
            
        usd_strength /= len(scores)
        
        result.score = float(usd_strength)
        result.features = {"global_usd_strength": float(usd_strength)}
        result.metadata = {"major_trends": scores}
        
        return result
