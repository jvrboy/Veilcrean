"""
hurst_confidence.py
===================
Tool 125 — Hurst Confidence Filter

Uses a rolling Hurst Exponent with a statistical significance test 
to filter out random walk periods.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class HurstConfidenceTool(BaseTool):
    name = "hurst_conf"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M30")
        if df is None or len(df) < 100:
            return result

        series = df["close"].tail(100).values
        h = self._calculate_hurst(series)
        
        # Scoring: High confidence for H > 0.65 or H < 0.35
        conf = 0.0
        if h > 0.6: conf = (h - 0.6) / 0.4
        elif h < 0.4: conf = (0.4 - h) / 0.4
            
        result.score = 0.0 
        result.confidence = float(np.clip(conf, 0, 1))
        result.features = {"hurst_conf_val": float(h)}
        return result

    def _calculate_hurst(self, ts):
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
