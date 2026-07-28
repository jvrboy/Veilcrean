"""
adr_filter.py
=============
Tool 19 — ADR (Average Daily Range) Exhaustion

Checks if the current daily range is exhausted, suggesting a reversal 
or a halt in the trend.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd

from .base_tool import BaseTool, ToolResult

class ADRFilterTool(BaseTool):
    name = "adr_filter"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        d1_df = buffers.get("D1")
        if d1_df is None or len(d1_df) < 5:
            return result

        adr = (d1_df["high"] - d1_df["low"]).rolling(5).mean().iloc[-1]
        
        # Current day's range
        # We might not have the full current day if it's just starting, 
        # so we estimate from the latest candles
        current_high = ctx.get("day_high", d1_df["high"].iloc[-1])
        current_low  = ctx.get("day_low", d1_df["low"].iloc[-1])
        current_range = current_high - current_low
        
        exhaustion = current_range / (adr + 1e-9)
        
        # If exhaustion > 0.9, we are nearing the average range. 
        # High probability of reversal or consolidation.
        
        result.score = 0.0 # ADR doesn't give direction, but limits it
        result.confidence = 1.0
        result.features = {"adr_exhaustion": float(exhaustion)}
        result.metadata = {"adr_pips": float(adr)}
        
        return result
