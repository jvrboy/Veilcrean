"""
price_volume_trend.py
======================
Tool 79 — Price Volume Trend (PVT)

Similar to OBV, but adjusts volume by the percentage price change, 
providing a more precise measure of money flow.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class PVTTool(BaseTool):
    name = "pvt"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 10:
            return result

        close = df["close"]
        vol = df["volume"] if "volume" in df.columns else (df["high"] - df["low"])
        
        pvt = (vol * (close.diff() / close.shift(1))).cumsum()
        
        pvt_delta = pvt.iloc[-1] - pvt.iloc[-10]
        
        result.score = float(np.tanh(pvt_delta / (vol.mean() + 1e-9)))
        result.features = {"pvt_delta": float(pvt_delta)}
        return result
