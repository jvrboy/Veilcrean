"""
awesome_oscillator.py
=====================
Tool 49 — Awesome Oscillator (AO)

Calculates the difference between a 34-period and 5-period Simple Moving 
Average of the median prices.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class AwesomeOscillatorTool(BaseTool):
    name = "awesome_osc"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 34:
            return result

        median_p = (df["high"] + df["low"]) / 2
        ao = median_p.rolling(5).mean() - median_p.rolling(34).mean()
        
        last_ao = ao.iloc[-1]
        prev_ao = ao.iloc[-2]
        
        # Color: Green if ao > prev_ao, Red if ao < prev_ao
        result.score = float(np.tanh(last_ao * 100))
        result.features = {
            "ao_val": float(last_ao),
            "ao_rising": float(1.0 if last_ao > prev_ao else -1.0)
        }
        return result
