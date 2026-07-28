"""
seasonal_tendency.py
====================
Tool 25 — Seasonal Tendency

Analyzes if the current Day of Week or Month has a historical bullish
or bearish bias for the asset.
"""
from __future__ import annotations
from datetime import datetime
from typing import Dict
import pandas as pd

from .base_tool import BaseTool, ToolResult

class SeasonalTendencyTool(BaseTool):
    name = "seasonal_tendency"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        now = ctx.get("now") or datetime.now()
        
        weekday = now.weekday() # 0=Mon
        month   = now.month
        
        # Historical biases (Example: GBP is often bullish in April, bearish in Sept)
        # In a real tool, these would be loaded from a CSV or DB
        # Here we use common Forex biases
        biases = {
            "EURUSD": { "months": {4: 0.4, 9: -0.3}, "days": {2: 0.2, 4: -0.1} },
            "GBPUSD": { "months": {4: 0.6, 5: -0.4}, "days": {1: 0.3} },
            "USDJPY": { "months": {1: -0.5, 3: 0.4} }
        }
        
        symbol = ctx.get("symbol", "")
        symbol_bias = biases.get(symbol, {})
        
        m_bias = symbol_bias.get("months", {}).get(month, 0.0)
        d_bias = symbol_bias.get("days", {}).get(weekday, 0.0)
        
        total_bias = m_bias + d_bias
        
        result.score = float(total_bias)
        result.confidence = 0.5
        result.features = {"seasonal_score": float(total_bias)}
        
        return result
