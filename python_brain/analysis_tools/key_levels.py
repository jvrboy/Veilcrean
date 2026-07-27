"""
key_levels.py
=============
Tool 5 — Key Levels & Fibonacci

Auto-detects:
    * Recent swing highs/lows
    * Round number / psychological levels
    * Fibonacci retracement levels from the last major swing
    * Previous day / week highs, lows, closes
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult
from ..config import TIMEFRAMES


class KeyLevelsTool(BaseTool):
    name = "key_levels"

    def __init__(self, pip_size: float = 0.0001):
        super().__init__()
        self.pip_size = pip_size

    # -------------------------------------------------------------- public
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        try:
            pip   = ctx.get("pip_size", self.pip_size)
            price = ctx.get("price")
            if price is None:
                df = buffers.get("H1") or buffers.get("M15")
                if df is not None and not df.empty:
                    price = float(df["close"].iloc[-1])
            if price is None:
                result.errors.append("no price")
                return result

            score = 0.0
            nearest: Optional[float] = None
            nearest_dist = float("inf")
            features: Dict[str, float] = {}

            # 1. Recent swing levels from H1
            h1 = buffers.get("H1")
            if h1 is not None and len(h1) >= 30:
                sh, sl = self._recent_swings(h1)
                for lvl in sh + sl:
                    d = abs(price - lvl)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest      = lvl
                features["kl_h1_levels"] = float(len(sh) + len(sl))

            # 2. Fibonacci of last major swing
            d1 = buffers.get("D1")
            if d1 is not None and len(d1) >= 30:
                sh, sl = self._recent_swings(d1)
                if sh and sl:
                    hi, lo = max(sh), min(sl)
                    if hi > lo:
                        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                        for f in fib_levels:
                            lvl = lo + (hi - lo) * f
                            d = abs(price - lvl)
                            if d < nearest_dist:
                                nearest_dist = d
                                nearest      = lvl
                        features["kl_fib_range_pips"] = (hi - lo) / pip

            # 3. Round numbers
            if self.pip_size > 0:
                round_step = 100 * pip
                nearest_round = round(price / round_step) * round_step
                if abs(price - nearest_round) < nearest_dist:
                    nearest_dist = abs(price - nearest_round)
                    nearest      = nearest_round
                features["kl_nearest_round_pips"] = abs(price - nearest_round) / pip

            # Score: price above nearest level = +1, below = -1 (cheap heuristic)
            if nearest is not None:
                if   price > nearest: score =  0.5
                elif price < nearest: score = -0.5
                else:                 score =  0.0
                # closer to level = higher confidence
                conf_boost = max(0.0, 1.0 - nearest_dist / (50 * pip))  # 50-pip window
                result.confidence = float(0.4 + 0.5 * conf_boost)

            result.score    = float(np.clip(score, -1, 1))
            result.features = features
            result.metadata = {"nearest_level": nearest, "distance_pips": nearest_dist / pip if nearest else None}
        except Exception as e:
            result.errors.append(f"key-levels failed: {e}")
        return result

    # -------------------------------------------------------------- internal
    @staticmethod
    def _recent_swings(df: pd.DataFrame, lookback: int = 3, count: int = 5):
        highs, lows = [], []
        h, l = df["high"].values, df["low"].values
        for i in range(lookback, len(h) - lookback):
            if h[i] == h[i-lookback:i+lookback+1].max():
                highs.append(float(h[i]))
            if l[i] == l[i-lookback:i+lookback+1].min():
                lows.append(float(l[i]))
        return highs[-count:], lows[-count:]
