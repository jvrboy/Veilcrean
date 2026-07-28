"""
chaikin_money_flow.py
=====================
Tool 43 — Chaikin Money Flow (CMF)

Measures the amount of Money Flow Volume over a specific period. 
Checks for accumulation/distribution.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class ChaikinMoneyFlowTool(BaseTool):
    name = "cmf"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 21:
            return result

        high = df["high"]
        low = df["low"]
        close = df["close"]
        vol = df["volume"] if "volume" in df.columns else (high - low)
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
        mfv = mfm * vol
        
        cmf = mfv.rolling(20).sum() / vol.rolling(20).sum()
        
        cmf_val = cmf.iloc[-1]
        
        result.score = float(np.tanh(cmf_val * 5))
        result.features = {"cmf_val": float(cmf_val)}
        return result
