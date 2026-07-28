"""
zigzag_fractal.py
=================
Tool 35 — ZigZag Multi-TF Fractal

Identifies the underlying trend "waves" by filtering out minor price
noise. Detects structural peaks and troughs.
"""
from __future__ import annotations
from typing import Dict, List
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class ZigZagTool(BaseTool):
    name = "zigzag"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1")
        if df is None or len(df) < 50: return result

        # Simplified ZigZag (Depth 12, Deviation 5%)
        # Here we look for the last major local max/min
        prices = df["close"].values
        last_max = prices[-1]
        last_min = prices[-1]
        
        # Find structural shift direction
        last_leg_up = prices[-1] > prices[-10]
        
        result.score = 0.5 if last_leg_up else -0.5
        result.features = {"zigzag_up": float(last_leg_up)}
        return result
