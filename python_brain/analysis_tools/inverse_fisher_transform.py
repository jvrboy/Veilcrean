"""
inverse_fisher_transform.py
===========================
Tool 119 — Inverse Fisher Transform (IFT)

Normalizes oscillators (like RSI) into a probability distribution 
that has clear boundaries at -1 and +1.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class InverseFisherTool(BaseTool):
    name = "inverse_fisher"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 14:
            return result

        # 1. RSI
        n = 14
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = -delta.clip(upper=0).rolling(n).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        # 2. Inverse Fisher Transform
        # v = 0.1 * (RSI - 50)
        v = 0.1 * (rsi - 50)
        # IFT = (EXP(2v)-1) / (EXP(2v)+1)
        ift = (np.exp(2 * v) - 1) / (np.exp(2 * v) + 1)
        
        last_ift = ift.iloc[-1]
        
        result.score = float(last_ift)
        result.features = {"ift_rsi": float(last_ift)}
        return result
