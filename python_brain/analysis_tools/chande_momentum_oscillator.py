"""
chande_momentum_oscillator.py
==============================
Tool 68 — Chande Momentum Oscillator (CMO)

Similar to RSI but uses price changes in both up and down directions 
directly in the numerator.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class CMOTool(BaseTool):
    name = "cmo"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 14:
            return result

        n = 14
        diff = df["close"].diff()
        su = diff.clip(lower=0).rolling(n).sum()
        sd = (-diff.clip(upper=0)).rolling(n).sum()
        
        cmo = 100 * (su - sd) / (su + sd + 1e-9)
        
        last_cmo = cmo.iloc[-1]
        
        result.score = float(last_cmo / 100.0)
        result.features = {"cmo_val": float(last_cmo)}
        return result
