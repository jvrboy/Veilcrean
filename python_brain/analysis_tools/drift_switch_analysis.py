"""
drift_switch_analysis.py
========================
Tool 28 — Drift Switch Index (DSI) Master Tool

Specialized tool for Deriv's Drift Switch Indices (10, 20, 30).
Analyzes "Drift" (sustained momentum) vs "Switch" (abrupt reversals).
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult

class DriftSwitchTool(BaseTool):
    name = "drift_switch_index"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M1")
        if df is None or len(df) < 50:
            return result

        # 1. Calculate Drift (Log returns mean)
        log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        if len(log_returns) < 30: return result
        
        # 2. Probability of Switch (based on Drift exhaustion)
        # In DSI, a 'switch' is more likely when drift has been sustained for X periods
        drift_window = 14
        drift = log_returns.rolling(window=drift_window).mean()
        
        # Standardize drift (Z-score)
        drift_z = (drift - drift.rolling(50).mean()) / (drift.rolling(50).std() + 1e-9)
        current_z = drift_z.iloc[-1]
        
        # 3. Volatility Spike Detection
        # Switches in DSI are often preceded by a decrease in volatility followed by a spike
        vol = log_returns.rolling(window=10).std()
        vol_ratio = vol.iloc[-1] / (vol.rolling(50).mean().iloc[-1] + 1e-9)
        
        score = 0.0
        # Over-extended Drift (Z-score > 2 or < -2) + Volatility starting to rise
        if current_z > 2.0 and vol_ratio > 1.2:
            score = -1.0 # High probability of Bearish Switch
        elif current_z < -2.0 and vol_ratio > 1.2:
            score = 1.0  # High probability of Bullish Switch
        else:
            # Continue following the drift
            score = float(np.tanh(current_z))

        result.score = score
        result.confidence = min(1.0, abs(current_z) / 3.0)
        result.features = {
            "dsi_z_score": float(current_z),
            "dsi_vol_ratio": float(vol_ratio),
            "dsi_drift_raw": float(drift.iloc[-1])
        }
        
        return result
