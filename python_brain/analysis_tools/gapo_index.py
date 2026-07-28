"""
gapo_index.py
=============
Tool 129 — Gopalakrishnan Range Index (GAPO)
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class GapOIndexTool(BaseTool):
    name = "gapo"
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20: return res
        h, l = df["high"].tail(14).max(), df["low"].tail(14).min()
        gapo = np.log10(h - l + 1e-9) / np.log10(14)
        res.features = {"gapo_val": float(gapo)}
        return res
