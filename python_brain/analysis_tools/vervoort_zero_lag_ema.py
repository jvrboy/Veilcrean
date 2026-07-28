"""
vervoort_zero_lag_ema.py
=========================
Tool 128 — Vervoort's Zero-Lag EMA
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class VervoortZeroLagTool(BaseTool):
    name = "vervoort_zl_ema"
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20: return res
        c = df["close"]
        e1 = c.ewm(span=14, adjust=False).mean()
        e2 = e1.ewm(span=14, adjust=False).mean()
        zl = e1 + (e1 - e2)
        res.score = float(np.tanh((zl.iloc[-1] - zl.iloc[-2]) * 1000))
        return res
