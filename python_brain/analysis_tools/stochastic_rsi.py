"""
stochastic_rsi.py
=================
Tool 81 — Stochastic RSI

Applies the Stochastic formula to RSI values instead of price, providing
a more sensitive indicator of overbought/oversold conditions.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class StochasticRSITool(BaseTool):
    name = "stoch_rsi"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 30:
            return result

        n = 14
        # 1. RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = -delta.clip(upper=0).rolling(n).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        # 2. Stochastic on RSI
        stoch_rsi = (rsi - rsi.rolling(n).min()) / (rsi.rolling(n).max() - rsi.rolling(n).min() + 1e-9)
        
        last_val = stoch_rsi.iloc[-1]
        
        result.score = float((last_val - 0.5) * 2)
        result.features = {"stoch_rsi_val": float(last_val)}
        return result
