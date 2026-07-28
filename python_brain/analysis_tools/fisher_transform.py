"""
fisher_transform.py
===================
Tool 53 — Fisher Transform

A technical indicator used to identify trend reversals by converting 
price into a Gaussian normal distribution.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class FisherTransformTool(BaseTool):
    name = "fisher"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 10:
            return result

        n = 10
        hl2 = (df["high"] + df["low"]) / 2
        hh = hl2.rolling(n).max()
        ll = hl2.rolling(n).min()
        
        # Value mapping to [-0.999, 0.999]
        val = 0.66 * ((hl2 - ll) / (hh - ll + 1e-9) - 0.5) + 0.67 * 0 # Simplified smoothing
        val = np.clip(val, -0.999, 0.999)
        
        fisher = 0.5 * np.log((1 + val) / (1 - val))
        # Simple smoothing
        fisher_smooth = fisher.rolling(3).mean()
        
        last_f = fisher_smooth.iloc[-1]
        
        result.score = float(np.tanh(last_f))
        result.features = {"fisher_val": float(last_f)}
        return result
