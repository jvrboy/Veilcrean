"""
market_profile_tpo.py
=====================
Tool 23 — Market Profile (Time Price Opportunity)

Calculates the Value Area and POC based on time spent at price levels.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class MarketProfileTool(BaseTool):
    name = "market_profile"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M30") or buffers.get("H1")
        if df is None or len(df) < 48: # 1 day of M30
            return result

        # Look at last 24 hours
        recent = df.tail(48)
        
        # We count how many periods each price level was touched
        # This is a proxy for TPO
        all_prices = []
        for _, row in recent.iterrows():
            # Create a range of prices for this bar
            p_range = np.linspace(row['low'], row['high'], 10)
            all_prices.extend(p_range)
            
        bins = np.linspace(min(all_prices), max(all_prices), 30)
        tpo_counts, _ = np.histogram(all_prices, bins=bins)
        
        tpo_poc_idx = np.argmax(tpo_counts)
        tpo_poc = (bins[tpo_poc_idx] + bins[tpo_poc_idx+1]) / 2.0
        
        price = ctx.get("price", recent["close"].iloc[-1])
        
        result.score = 0.0 # Neutral, used for confluence
        result.features = {
            "tpo_poc_dist": float(price - tpo_poc),
            "tpo_poc": float(tpo_poc)
        }
        result.metadata = {"tpo_poc": tpo_poc}
        return result
