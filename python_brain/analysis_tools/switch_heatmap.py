"""
switch_heatmap.py
=================
Tool 29 — Switch Probability Heatmap

Calculates a probability distribution of when the next 'Switch' will occur
based on the elapsed time since the last drift change.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from .base_tool import BaseTool, ToolResult

class SwitchHeatmapTool(BaseTool):
    name = "switch_heatmap"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M1")
        if df is None or len(df) < 50: return result
        
        # 1. Detect last switch
        log_ret = np.log(df["close"] / df["close"].shift(1))
        # Drift direction: sign of mean log return
        drift_dir = np.sign(log_ret.rolling(14).mean())
        
        # Find where drift_dir changed
        switches = drift_dir.diff().fillna(0) != 0
        if not switches.any():
            return result
            
        last_switch_idx = df.index[switches][-1]
        bars_since_switch = len(df) - df.index.get_loc(last_switch_idx)
        
        # 2. Probability Curve
        # DSI switches often happen in clusters or after specific durations
        # Based on DSI 10/20/30 behavior, switches happen roughly every 30-100 bars on M1
        mean_bars = 60
        std_bars  = 20
        
        # Cumulative Distribution Function (Normal Dist)
        from scipy.stats import norm
        prob = norm.cdf(bars_since_switch, loc=mean_bars, scale=std_bars)
        
        result.score = 0.0 # Neutral, providing probability feature
        result.confidence = 1.0
        result.features = {
            "switch_probability": float(prob),
            "bars_since_last_switch": float(bars_since_switch)
        }
        result.metadata = {"prob_pct": f"{prob*100:.1f}%"}
        
        return result
