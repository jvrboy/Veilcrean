"""
ulcer_index.py
==============
Tool 62 — Ulcer Index

Measures the depth and duration of price declines from earlier highs. 
Focuses on "Stress" and "Drawdown Risk."
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class UlcerIndexTool(BaseTool):
    name = "ulcer_index"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 14:
            return result

        n = 14
        close = df["close"]
        max_close = close.rolling(n).max()
        
        # Percentage drawdown
        drawdown = 100 * (close - max_close) / max_close
        # Squared average of drawdowns
        ui = np.sqrt((drawdown**2).rolling(n).mean())
        
        last_ui = ui.iloc[-1]
        
        result.score = 0.0 # Indicator of risk, not direction
        result.confidence = 1.0
        result.features = {"ulcer_index": float(last_ui)}
        result.metadata = {"risk_level": "HIGH" if last_ui > 5 else "LOW"}
        
        return result
