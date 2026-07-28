"""
vertical_horizontal_filter.py
==============================
Tool 70 — Vertical Horizontal Filter (VHF)

Identifies whether price is in a trending phase or a congestion phase.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class VHFTool(BaseTool):
    name = "vhf"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 28:
            return result

        n = 28
        close = df["close"]
        h = close.rolling(n).max()
        l = close.rolling(n).min()
        
        num = (h - l).abs()
        den = close.diff().abs().rolling(n).sum()
        
        vhf = num / (den + 1e-9)
        
        last_vhf = vhf.iloc[-1]
        
        result.score = 0.0 # Regime indicator
        result.features = {"vhf_val": float(last_vhf)}
        result.metadata = {"market_state": "TRENDING" if last_vhf > 0.4 else "RANGING"}
        return result
