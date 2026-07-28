"""
on_balance_volume.py
====================
Tool 48 — On-Balance Volume (OBV)

Uses volume flow to predict changes in stock price. Accumulates volume 
on up days and subtracts it on down days.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class OBVTool(BaseTool):
    name = "obv"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 10:
            return result

        close = df["close"]
        vol = df["volume"] if "volume" in df.columns else (df["high"] - df["low"])
        
        obv = (np.sign(close.diff()).fillna(0) * vol).cumsum()
        
        # Scoring: slope of OBV over last 10 bars
        obv_delta = obv.iloc[-1] - obv.iloc[-10]
        
        result.score = float(np.tanh(obv_delta / (vol.mean() * 5 + 1e-9)))
        result.features = {"obv_delta": float(obv_delta)}
        return result
