"""
vwap_anchored.py
================
Tool 32 — Anchored VWAP

Calculates VWAP from a specific anchor point (e.g., Session Start, 
Weekly High, or Major News event).
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class AnchoredVWAPTool(BaseTool):
    name = "anchored_vwap"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 50:
            return result

        # Anchor: Start of the current day (simplified)
        # In a full version, we'd detect the day boundary in the dataframe
        anchor_idx = len(df) - min(len(df), 48) # Last 12 hours approx
        
        subset = df.iloc[anchor_idx:]
        
        typ_price = (subset["high"] + subset["low"] + subset["close"]) / 3
        vol = subset["volume"] if "volume" in subset.columns else (subset["high"] - subset["low"])
        
        vwap = (typ_price * vol).cumsum() / vol.cumsum()
        
        last_vwap = vwap.iloc[-1]
        last_close = df["close"].iloc[-1]
        
        score = 0.0
        if last_close > last_vwap:
            score = 0.6 # Bullish above VWAP
        else:
            score = -0.6 # Bearish below VWAP
            
        result.score = score
        result.features = {
            "vwap_dist": float(last_close - last_vwap),
            "vwap_pos": float(1.0 if last_close > last_vwap else -1.0)
        }
        result.metadata = {"vwap": last_vwap}
        
        return result
