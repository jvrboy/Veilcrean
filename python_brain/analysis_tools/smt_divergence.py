"""
smt_divergence.py
=================
Tool 13 — SMT (Smart Money Technique) Divergence

Detects when correlated pairs (e.g. EURUSD and GBPUSD) fail to make
symmetrical highs/lows, indicating institutional manipulation.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class SMTDivergenceTool(BaseTool):
    name = "smt_divergence"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        symbol = ctx.get("symbol", "")
        
        # Correlated pair lookup
        correlates = {
            "EURUSD": "GBPUSD",
            "GBPUSD": "EURUSD",
            "AUDUSD": "NZDUSD",
            "NZDUSD": "AUDUSD",
            "USDJPY": "USDCHF",
        }
        
        target = correlates.get(symbol)
        if not target: return result
        
        main_df = buffers.get("H1")
        corr_df = buffers.get(f"{target}_H1")
        
        if main_df is None or corr_df is None or len(main_df) < 2 or len(corr_df) < 2:
            return result
            
        # Bullish SMT: One makes LL, other makes HL
        # Bearish SMT: One makes HH, other makes LH
        
        main_h = main_df["high"].iloc[-1]
        main_prev_h = main_df["high"].iloc[-2]
        corr_h = corr_df["high"].iloc[-1]
        corr_prev_h = corr_df["high"].iloc[-2]
        
        score = 0.0
        # Bearish SMT
        if (main_h > main_prev_h and corr_h < corr_prev_h) or \
           (main_h < main_prev_h and corr_h > corr_prev_h):
            score = -0.5 # Reversal signal
            
        result.score = score
        result.confidence = 0.7
        result.features = {"smt_div_found": float(score != 0)}
        
        return result
