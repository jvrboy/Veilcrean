"""
tick_urgency.py
===============
Tool 21 — Tick Speed & Urgency

Analyzes the frequency and size of incoming ticks to detect "Institutional Urgency"
or high-frequency activity before a breakout.
"""
from __future__ import annotations
from typing import Dict, List
import time
import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult

class TickUrgencyTool(BaseTool):
    name = "tick_urgency"

    def __init__(self):
        super().__init__()
        self.tick_timestamps: List[float] = []
        self.tick_volumes: List[float] = []

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        
        # We need the last tick metadata
        tick_data = ctx.get("tick")
        if not tick_data:
            return result
            
        now = time.time()
        self.tick_timestamps.append(now)
        # Assuming tick volume or spread as a proxy for activity
        self.tick_volumes.append(ctx.get("spread", 1.0))
        
        # Keep only last 60 seconds
        cutoff = now - 60
        while self.tick_timestamps and self.tick_timestamps[0] < cutoff:
            self.tick_timestamps.pop(0)
            self.tick_volumes.pop(0)
            
        if len(self.tick_timestamps) < 5:
            return result
            
        # Ticks per second
        tps = len(self.tick_timestamps) / 60.0
        
        # Acceleration: is the frequency increasing?
        if len(self.tick_timestamps) > 20:
            recent_tps = 10 / (self.tick_timestamps[-1] - self.tick_timestamps[-10])
            avg_tps = len(self.tick_timestamps) / (self.tick_timestamps[-1] - self.tick_timestamps[0])
            urgency = recent_tps / (avg_tps + 1e-9)
        else:
            urgency = 1.0
            
        result.score = float(np.tanh(urgency - 1.0))
        result.confidence = 0.6
        result.features = {
            "ticks_per_sec": float(tps),
            "tick_urgency":  float(urgency)
        }
        return result
