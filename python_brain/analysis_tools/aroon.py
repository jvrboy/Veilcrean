"""
aroon.py
========
Tool 54 — Aroon Indicator

Measures the time between highs and the time between lows over a period 
to identify trend strength and direction.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class AroonTool(BaseTool):
    name = "aroon"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 25:
            return result

        n = 25
        aroon_up = df["high"].rolling(n + 1).apply(lambda x: float(np.argmax(x)) / n * 100, raw=True)
        aroon_down = df["low"].rolling(n + 1).apply(lambda x: float(np.argmin(x)) / n * 100, raw=True)
        
        last_up = aroon_up.iloc[-1]
        last_down = aroon_down.iloc[-1]
        
        # Scoring: up - down
        result.score = float((last_up - last_down) / 100.0)
        result.features = {
            "aroon_up": float(last_up),
            "aroon_down": float(last_down)
        }
        return result
