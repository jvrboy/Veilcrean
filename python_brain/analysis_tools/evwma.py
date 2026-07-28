"""
evwma.py
========
Tool 126 — Elastic Volume Weighted Moving Average (EVWMA)
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class EVWMATool(BaseTool):
    name = "evwma"
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20: return res
        n = 20
        vol_float = df["volume"].rolling(n * 5).sum().iloc[-1]
        close, vol = df["close"].values, df["volume"].values
        ev = np.zeros_like(close); ev[0] = close[0]
        for i in range(1, len(close)):
            ev[i] = ((vol_float - vol[i]) * ev[i-1] + vol[i] * close[i]) / (vol_float + 1e-9)
        res.score = float(np.tanh((close[-1] - ev[-1]) / (close[-1] * 0.01 + 1e-9)))
        res.features = {"evwma_dist": res.score}
        return res
