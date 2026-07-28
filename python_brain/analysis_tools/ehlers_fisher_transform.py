"""
ehlers_fisher_transform.py
==========================
Tool 99 — Ehlers Advanced Fisher Transform

An improved version of the Fisher Transform by John Ehlers, using pre-smoothing
to reduce noise and false signals.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class EhlersFisherTool(BaseTool):
    name = "ehlers_fisher"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 15:
            return result

        n = 10
        hl2 = (df["high"] + df["low"]) / 2
        
        # Pre-smoothing
        smoothed = hl2.rolling(3).mean()
        
        hh = smoothed.rolling(n).max()
        ll = smoothed.rolling(n).min()
        
        # Value mapping to [-0.999, 0.999]
        val = 0.33 * 2 * ((smoothed - ll) / (hh - ll + 1e-9) - 0.5) + 0.67 * 0 # Recursive simplified
        val = np.clip(val, -0.999, 0.999)
        
        fisher = 0.5 * np.log((1 + val) / (1 - val))
        # Simple smoothing
        fisher_smooth = fisher.rolling(3).mean()
        
        last_f = fisher_smooth.iloc[-1]
        
        result.score = float(np.tanh(last_f))
        result.features = {"ehlers_fisher_val": float(last_f)}
        return result
