"""
relative_volatility_index.py
============================
Tool 34 — Relative Volatility Index (RVI)

Similar to RSI but measures the direction of volatility (Standard Deviation).
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class RVITool(BaseTool):
    name = "rvi"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        period = 14
        change = df["close"].diff()
        
        # SD when price went up vs when price went down
        up = pd.Series(np.where(change > 0, df["close"].rolling(10).std(), 0))
        down = pd.Series(np.where(change < 0, df["close"].rolling(10).std(), 0))
        
        avg_up = up.rolling(period).mean()
        avg_down = down.rolling(period).mean()
        
        rvi = 100 * (avg_up / (avg_up + avg_down + 1e-9))
        
        rvi_val = rvi.iloc[-1]
        
        result.score = float((rvi_val - 50) / 50)
        result.confidence = 0.6
        result.features = {"rvi_score": result.score}
        return result
