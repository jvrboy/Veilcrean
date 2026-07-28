"""
order_flow_imbalance.py
=======================
Tool 26 — Order Flow Imbalance (Delta)

Analyzes the difference between aggressive buying and aggressive selling
volume to detect institutional pressure.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult

class OrderFlowImbalanceTool(BaseTool):
    name = "order_flow"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M1")
        if df is None or len(df) < 5:
            return result

        # In standard OHLCV, we don't have separate Buy/Sell volume, 
        # so we use price action + volume as a proxy for delta.
        # (Close > Open) -> Bullish volume
        # (Close < Open) -> Bearish volume
        
        recent = df.tail(10)
        bull_vol = recent[recent["close"] > recent["open"]]["volume"].sum()
        bear_vol = recent[recent["close"] < recent["open"]]["volume"].sum()
        
        delta = bull_vol - bear_vol
        total_vol = bull_vol + bear_vol
        imbalance = delta / (total_vol + 1e-9)
        
        result.score = float(np.tanh(imbalance * 3))
        result.confidence = 0.7
        result.features = {"volume_delta": float(imbalance)}
        result.metadata = {"bull_vol": float(bull_vol), "bear_vol": float(bear_vol)}
        
        return result
