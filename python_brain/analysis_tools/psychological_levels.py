"""
psychological_levels.py
========================
Tool 71 — Psychological Levels (Round Numbers)

Identifies major round numbers (e.g., 1.1000, 1.1050) which act as 
natural institutional support and resistance.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult

class PsychologicalLevelsTool(BaseTool):
    name = "psych_levels"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        price = ctx.get("price")
        if price is None: return result

        # Define 'roundness' based on the asset price scale
        # For FX (1.1234), round levels are 0.0100 (Big Figures) and 0.0050 (Mid Figures)
        # For Crypto (50000), round levels are 1000 and 500
        
        if price > 1000:
            base = 500
        elif price > 10:
            base = 1.0
        else:
            base = 0.01

        nearest_level = round(price / base) * base
        dist = (price - nearest_level) / price
        
        # Scoring: Reversal bias when very close to a round number
        result.score = -float(np.tanh(dist * 500)) 
        result.confidence = 0.5
        result.features = {"dist_to_round_num": float(dist)}
        result.metadata = {"nearest_level": nearest_level}
        
        return result
