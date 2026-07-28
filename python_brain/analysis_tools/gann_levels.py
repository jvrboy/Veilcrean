"""
gann_levels.py
==============
Tool 39 — Gann Geometric Levels

Calculates support and resistance based on the Square of Nine and 
geometric angles.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class GannLevelTool(BaseTool):
    name = "gann_levels"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        price = ctx.get("price")
        if price is None: return result

        # Simplified Gann Square of Nine: 
        # root = sqrt(price), then add degrees
        root = np.sqrt(price)
        
        # S/R levels at 45, 90, 180 degrees
        levels = []
        for deg in [45, 90, 180, 270, 360]:
            levels.append((root + deg/180)**2)
            levels.append((root - deg/180)**2)
            
        # Find nearest level
        closest = min(levels, key=lambda x: abs(x - price))
        dist = (price - closest) / price
        
        result.score = -float(np.tanh(dist * 100)) # Reversal near Gann levels
        result.confidence = 0.5
        result.features = {"dist_to_gann": float(dist)}
        return result
