"""
market_heat_index.py
====================
Tool 110 — Market Heat Index

Combines normalized volume, volatility, and momentum into a single 
'Activity' score to identify high-interest areas.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class MarketHeatTool(BaseTool):
    name = "market_heat"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 20:
            return result

        # 1. Normalized Volume
        vol = df["volume"] if "volume" in df.columns else (df["high"] - df["low"])
        vol_n = vol / vol.rolling(20).mean()
        
        # 2. Normalized ATR
        tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs()], axis=1).max(axis=1)
        atr_n = tr / tr.rolling(20).mean()
        
        # 3. Momentum
        mom = df["close"].diff().abs() / df["close"].shift(1)
        mom_n = mom / mom.rolling(20).mean()
        
        heat = (vol_n.iloc[-1] + atr_n.iloc[-1] + mom_n.iloc[-1]) / 3
        
        result.score = 0.0 # Heat is an intensity measure
        result.features = {"market_heat": float(heat)}
        result.metadata = {"heat_level": "EXTREME" if heat > 2.0 else "NORMAL"}
        return result
