"""
coppock_curve.py
================
Tool 55 — Coppock Curve

A momentum-based indicator used primarily to identify long-term buying 
opportunities.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class CoppockCurveTool(BaseTool):
    name = "coppock"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H4") or buffers.get("H1")
        if df is None or len(df) < 30:
            return result

        close = df["close"]
        roc14 = ((close - close.shift(14)) / close.shift(14)) * 100
        roc11 = ((close - close.shift(11)) / close.shift(11)) * 100
        
        sum_roc = roc14 + roc11
        
        # WMA(10)
        def wma(s, period):
            weights = np.arange(1, period + 1)
            return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
            
        coppock = wma(sum_roc, 10)
        
        last_c = coppock.iloc[-1]
        
        result.score = float(np.tanh(last_c / 10.0))
        result.features = {"coppock_val": float(last_c)}
        return result
