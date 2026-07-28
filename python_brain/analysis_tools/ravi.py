"""
ravi.py
=======
Tool 76 — Range Action Verification Index (RAVI)

Identifies whether the market is in a trend or a range by comparing 
short-term and long-term moving averages.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class RAVITool(BaseTool):
    name = "ravi"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 65:
            return result

        # Formula: ABS( SMA(7) - SMA(65) ) / SMA(65) * 100
        sma7 = df["close"].rolling(7).mean()
        sma65 = df["close"].rolling(65).mean()
        
        ravi = (sma7 - sma65).abs() / (sma65 + 1e-9) * 100
        
        last_ravi = ravi.iloc[-1]
        
        result.score = 0.0 # Indicator of trend quality
        result.features = {"ravi_val": float(last_ravi)}
        result.metadata = {"is_trending": last_ravi > 3.0}
        
        return result
