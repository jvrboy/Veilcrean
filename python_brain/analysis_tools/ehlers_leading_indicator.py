"""
ehlers_leading_indicator.py
===========================
Tool 144 — Ehlers Leading Indicator

A specialized leading indicator created by John Ehlers that aims to 
anticipate turning points using advanced filtering.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from .base_tool import BaseTool, ToolResult

class LeadingIndicatorTool(BaseTool):
    name = "leading_indicator"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        res = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 20: return res

        prices = (df["high"] + df["low"]) / 2
        alpha1 = 0.25
        alpha2 = 0.33
        
        lead = np.zeros_like(prices)
        net_lead = np.zeros_like(prices)
        
        for i in range(2, len(prices)):
            lead[i] = 2 * prices.iloc[i] + (alpha1 - 2) * prices.iloc[i-1] + (1 - alpha1) * lead[i-1]
            net_lead[i] = alpha2 * lead[i] + (1 - alpha2) * net_lead[i-1]
            
        last_l = net_lead[-1]
        prev_l = net_lead[-2]
        
        res.score = float(np.tanh((last_l - prev_l) * 1000))
        res.features = {"leading_val": float(last_l)}
        return res
