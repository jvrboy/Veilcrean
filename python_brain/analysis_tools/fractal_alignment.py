"""
fractal_alignment.py
====================
Tool 14 — Fractal Alignment Confluence

Looks for Lower Timeframe (LTF) confirmation inside Higher Timeframe (HTF) zones.
Example: Price touches an H4 Order Block AND we see an M5 CHoCH.
"""
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd

from .base_tool import BaseTool, ToolResult
from .market_structure import MarketStructureTool
from .supply_demand import SupplyDemandTool

class FractalAlignmentTool(BaseTool):
    name = "fractal_alignment"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        
        # We need access to the results of other tools (or we run them here)
        # For simplicity, we'll check the buffers directly
        
        # 1. Check HTF (H4 or D1) for "In Zone"
        h4_df = buffers.get("H4")
        m5_df = buffers.get("M5")
        
        if h4_df is None or m5_df is None or len(h4_df) < 50 or len(m5_df) < 50:
            return result

        # Simplified check: Is M5 price near H4 swing points?
        h4_high = h4_df["high"].tail(20).max()
        h4_low  = h4_df["low"].tail(20).min()
        current_p = ctx.get("price", m5_df["close"].iloc[-1])
        
        # 2. Check LTF (M5) for BOS/CHoCH
        # (Re-using logic from MarketStructure would be better, but let's do a quick check)
        m5_close = m5_df["close"]
        m5_chock_bull = m5_close.iloc[-1] > m5_df["high"].iloc[-5:-1].max()
        m5_chock_bear = m5_close.iloc[-1] < m5_df["low"].iloc[-5:-1].min()

        score = 0.0
        # Bullish Fractal Alignment: At H4 Low + M5 Bullish break
        if abs(current_p - h4_low) / h4_low < 0.002 and m5_chock_bull:
            score = 1.0
        # Bearish Fractal Alignment: At H4 High + M5 Bearish break
        elif abs(current_p - h4_high) / h4_high < 0.002 and m5_chock_bear:
            score = -1.0
            
        result.score = score
        result.confidence = 0.8
        result.features = {"fractal_aligned": float(score != 0)}
        result.metadata = {"htf_range": [h4_low, h4_high]}
        
        return result
