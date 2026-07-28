"""
vhf_slope.py
============
Tool 130 — Vertical Horizontal Filter (VHF) Slope
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class VHFSlopeTool(BaseTool):
    name = "vhf_slope"
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 30: return res
        c = df["close"]; n = 28
        h, l = c.rolling(n).max(), c.rolling(n).min()
        vhf = (h - l).abs() / (c.diff().abs().rolling(n).sum() + 1e-9)
        slope = vhf.diff(3) / 3.0
        res.score = float(np.tanh(slope.iloc[-1] * 100))
        return res
