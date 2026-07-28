"""
herrick_payoff_index.py
========================
Tool 127 — Herrick Payoff Index (HPI)
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class HPITool(BaseTool):
    name = "hpi"
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 10: return res
        h, l, v = df["high"], df["low"], df["volume"]
        mp = (h + l) / 2
        hpi = (v * mp.diff()).rolling(10).mean()
        res.score = float(np.tanh(hpi.iloc[-1] / (v.mean() * 0.001 + 1e-9)))
        return res
