"""
aroon_oscillator.py
===================
Tool 83 — Aroon Oscillator

The difference between Aroon Up and Aroon Down. Ranges from -100 to +100.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class AroonOscillatorTool(BaseTool):
    name = "aroon_osc"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 25:
            return result

        n = 25
        up = df["high"].rolling(n + 1).apply(lambda x: float(np.argmax(x)) / n * 100, raw=True)
        down = df["low"].rolling(n + 1).apply(lambda x: float(np.argmin(x)) / n * 100, raw=True)
        
        osc = up - down
        last_osc = osc.iloc[-1]
        
        result.score = float(last_osc / 100.0)
        result.features = {"aroon_osc_val": float(last_osc)}
        return result
