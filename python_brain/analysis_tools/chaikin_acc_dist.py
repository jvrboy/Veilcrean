"""
chaikin_acc_dist.py
====================
Tool 149 — Chaikin Accumulation Distribution

A volume-based indicator that identifies whether a stock is being 
accumulated or distributed by measuring the location of the close 
relative to the high-low range.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class ChaikinADTool(BaseTool):
    name = "chaikin_ad"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None: return res

        high, low, close = df["high"], df["low"], df["close"]
        vol = df["volume"] if "volume" in df.columns else (high - low)
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
        ad = (mfm * vol).cumsum()
        
        # Slope of AD over last 10 bars
        slope = (ad.iloc[-1] - ad.iloc[-10]) / (vol.mean() * 10 + 1e-9)
        
        res.score = float(np.tanh(slope))
        res.features = {"cad_val": float(ad.iloc[-1]), "cad_slope": res.score}
        return res
