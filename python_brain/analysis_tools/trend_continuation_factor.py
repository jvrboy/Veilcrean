"""
trend_continuation_factor.py
============================
Tool 111 — Trend Continuation Factor (TCF)

Identifies the presence and direction of a trend. It distinguishes between 
trending and non-trending markets.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class TCFTool(BaseTool):
    name = "tcf"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 35:
            return result

        n = 35
        diff = df["close"].diff()
        
        change = diff
        plus_change = np.where(change > 0, change, 0)
        minus_change = np.where(change < 0, -change, 0)
        
        # Positive TCF
        plus_cf = np.zeros_like(plus_change)
        for i in range(1, len(plus_change)):
            plus_cf[i] = plus_change[i] + (plus_cf[i-1] if plus_change[i] > 0 else 0)
        
        # Negative TCF
        minus_cf = np.zeros_like(minus_change)
        for i in range(1, len(minus_change)):
            minus_cf[i] = minus_change[i] + (minus_cf[i-1] if minus_change[i] > 0 else 0)
            
        plus_tcf = pd.Series(plus_change - plus_cf).rolling(n).sum()
        minus_tcf = pd.Series(minus_change - minus_cf).rolling(n).sum()
        
        last_p = plus_tcf.iloc[-1]
        last_m = minus_tcf.iloc[-1]
        
        result.score = float(np.tanh((last_p - last_m) / (df["close"].std() * 10 + 1e-9)))
        result.features = {"tcf_plus": float(last_p), "tcf_minus": float(last_m)}
        return result
