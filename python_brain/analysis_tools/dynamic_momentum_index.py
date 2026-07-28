"""
dynamic_momentum_index.py
=========================
Tool 147 — Dynamic Momentum Index (DYMI)

A variable-length RSI that automatically adjusts its period based on 
market volatility.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class DYMITool(BaseTool):
    name = "dymi"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 30: return res

        close = df["close"]
        std = close.rolling(5).std()
        avg_std = std.rolling(30).mean()
        
        # Adaptive period
        k = std / (avg_std + 1e-9)
        period = (14 / k).fillna(14).clip(5, 30).astype(int)
        
        # Calculate RSI with adaptive period for last bar
        n = int(period.iloc[-1])
        delta = close.diff()
        gain = delta.clip(lower=0).tail(n).mean()
        loss = -delta.clip(upper=0).tail(n).mean()
        rs = gain / (loss + 1e-9)
        dymi = 100 - (100 / (1 + rs))
        
        res.score = float((dymi - 50) / 50)
        res.features = {"dymi_val": float(dymi)}
        return res
