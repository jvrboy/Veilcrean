"""
volume_profile.py
=================
Tool 12 — Volume Profile / POC (Point of Control)

Calculates the price level with the highest volume in the last N bars.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult

class VolumeProfileTool(BaseTool):
    name = "volume_profile"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 50:
            return result

        # Simplified Volume Profile: use 20 bins
        lookback = 100
        recent = df.tail(lookback)
        
        # Use tick volume if available, else range
        vol = recent["volume"] if "volume" in recent.columns else (recent["high"] - recent["low"])
        prices = recent["close"]
        
        min_p = prices.min()
        max_p = prices.max()
        if max_p == min_p: return result

        bins = np.linspace(min_p, max_p, 20)
        v_profile, _ = np.histogram(prices, bins=bins, weights=vol)
        
        poc_idx = np.argmax(v_profile)
        poc_price = (bins[poc_idx] + bins[poc_idx+1]) / 2.0
        
        current_price = ctx.get("price", prices.iloc[-1])
        
        # Score: +1 if price is bouncing off POC from above, -1 from below
        dist_to_poc = current_price - poc_price
        
        result.score = float(np.tanh(dist_to_poc * 100)) # Simple normalized distance
        result.features = {
            "poc_dist": float(dist_to_poc),
            "vol_at_poc": float(v_profile[poc_idx])
        }
        result.metadata = {"poc": poc_price}
        
        return result
