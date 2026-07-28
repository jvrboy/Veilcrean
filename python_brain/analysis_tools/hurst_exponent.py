"""
hurst_exponent.py
=================
Tool 22 — Hurst Exponent

A mathematical measure of long-term memory of time series.
H < 0.5: Mean-reverting (ranging)
H = 0.5: Random walk
H > 0.5: Trending
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult

class HurstExponentTool(BaseTool):
    name = "hurst_exponent"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 100:
            return result

        series = df["close"].tail(100).values
        h = self._calculate_hurst(series)
        
        result.score = 0.0 # Hurst is a regime indicator, not direction
        result.confidence = 1.0
        result.features = {"hurst_exponent": float(h)}
        
        # Guide the regime:
        regime_guess = "RANDOM"
        if h < 0.4: regime_guess = "RANGING"
        elif h > 0.6: regime_guess = "TRENDING"
        
        result.metadata = {"hurst": h, "regime_guess": regime_guess}
        return result

    def _calculate_hurst(self, ts):
        """Returns the Hurst Exponent of the time series."""
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
