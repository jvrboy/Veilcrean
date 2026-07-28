"""
woodies_cci.py
==============
Tool 118 — Woodies CCI

Uses two different periods of CCI (Turbo and Trend) to identify market 
momentum and high-probability patterns like ZL-Bounce.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class WoodiesCCITool(BaseTool):
    name = "woodies_cci"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20:
            return result

        tp = (df["high"] + df["low"] + df["close"]) / 3
        
        def cci(period):
            sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            return (tp - sma) / (0.015 * mad + 1e-9)
            
        cci_trend = cci(14)
        cci_turbo = cci(6)
        
        last_trend = cci_trend.iloc[-1]
        last_turbo = cci_turbo.iloc[-1]
        
        result.score = float(np.tanh((last_trend + last_turbo) / 200.0))
        result.features = {
            "cci_trend": float(last_trend),
            "cci_turbo": float(last_turbo)
        }
        return result
