"""
klinger_oscillator.py
=====================
Tool 47 — Klinger Volume Oscillator (KVO)

An oscillator that identifies long-term trends of money flow while 
remaining sensitive to short-term fluctuations.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np

from .base_tool import BaseTool, ToolResult

class KlingerOscillatorTool(BaseTool):
    name = "klinger"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("H1") or buffers.get("M15")
        if df is None or len(df) < 55:
            return result

        tp = (df["high"] + df["low"] + df["close"]) / 3
        sv = pd.Series(np.where(tp > tp.shift(1), df["volume"], -df["volume"]))
        
        kvo = sv.ewm(span=34).mean() - sv.ewm(span=55).mean()
        signal = kvo.ewm(span=13).mean()
        
        last_kvo = kvo.iloc[-1]
        last_sig = signal.iloc[-1]
        
        result.score = float(np.tanh((last_kvo - last_sig) / (df["volume"].mean() + 1e-9)))
        result.features = {"kvo_val": float(last_kvo)}
        return result
