"""
money_flow_index.py
===================
Tool 40 — Money Flow Index (MFI)

A momentum indicator that measures the inflow and outflow of money 
using both price and volume.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class MFITool(BaseTool):
    name = "mfi"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 15:
            return result

        tp = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"] if "volume" in df.columns else (df["high"] - df["low"])
        mf = tp * vol
        
        diff = tp.diff()
        pos_mf = pd.Series(np.where(diff > 0, mf, 0)).rolling(14).sum()
        neg_mf = pd.Series(np.where(diff < 0, mf, 0)).rolling(14).sum()
        
        mfr = pos_mf / (neg_mf + 1e-9)
        mfi = 100 - (100 / (1 + mfr))
        
        mfi_val = mfi.iloc[-1]
        
        result.score = float((mfi_val - 50) / 50)
        result.features = {"mfi_val": float(mfi_val)}
        return result
