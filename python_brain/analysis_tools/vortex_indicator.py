"""
vortex_indicator.py
===================
Tool 57 — Vortex Indicator (VI)

Consists of two lines (VI+ and VI-) that capture trend direction and 
strength by measuring the distance between current and previous price ranges.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class VortexTool(BaseTool):
    name = "vortex"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 15:
            return result

        n = 14
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1).rolling(n).sum()
        
        vmp = (high - low.shift()).abs().rolling(n).sum()
        vmm = (low - high.shift()).abs().rolling(n).sum()
        
        vip = vmp / (tr + 1e-9)
        vim = vmm / (tr + 1e-9)
        
        last_vip = vip.iloc[-1]
        last_vim = vim.iloc[-1]
        
        # Scoring: vip - vim
        result.score = float(np.clip(last_vip - last_vim, -1, 1))
        result.features = {
            "vortex_vip": float(last_vip),
            "vortex_vim": float(last_vim)
        }
        return result
