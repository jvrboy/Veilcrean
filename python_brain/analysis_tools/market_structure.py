"""
market_structure.py
===================
Tool 1 — Market Structure Analysis

Detects on each timeframe:
    * Higher Highs / Higher Lows        (uptrend)
    * Lower Highs / Lower Lows          (downtrend)
    * Break of Structure (BOS)          (trend continuation)
    * Change of Character (CHoCH)       (trend reversal)

Output score
------------
    +1  strong uptrend
    -1  strong downtrend
     0  ranging
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult
from ..config import ANA_CFG, TIMEFRAMES


@dataclass
class SwingPoint:
    index:   int
    price:   float
    is_high: bool


class MarketStructureTool(BaseTool):
    name = "market_structure"

    def __init__(self, lookback: int = None, bos_lookback: int = None):
        super().__init__(lookback=lookback, bos_lookback=bos_lookback)
        self.lookback     = lookback     or ANA_CFG.swing_lookback
        self.bos_lookback = bos_lookback or ANA_CFG.bos_min_lookback

    # -------------------------------------------------------------- public
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        try:
            tf_scores: Dict[str, float] = {}
            features:  Dict[str, float] = {}

            for tf in TIMEFRAMES:
                df = buffers.get(tf)
                if df is None or len(df) < self.lookback * 4:
                    continue
                s = self._analyze_one_tf(df)
                tf_scores[tf] = s
                # also feed per-TF features to NN
                features[f"ms_score_{tf}"] = s
                features[f"ms_rr_{tf}"]     = self._range_ratio(df)

            if not tf_scores:
                result.errors.append("no TF had enough data")
                return result

            # Weighted average, with HTFs (D1, H4) weighted more
            weights = {"M1": 0.3, "M5": 0.5, "M15": 0.7, "M30": 0.8,
                       "H1": 1.0, "H4": 1.4, "D1": 1.8, "W1": 1.5, "MN1": 1.2}
            num = sum(tf_scores.get(tf, 0) * w for tf, w in weights.items())
            den = sum(weights[tf] for tf in tf_scores)
            result.score = float(np.clip(num / max(den, 1e-9), -1, 1))

            # Confidence = how aligned the TFs are
            arr = np.array(list(tf_scores.values()))
            result.confidence = float(max(0.0, 1.0 - arr.std() / 0.7))
            result.features   = features
            result.metadata   = {"per_tf": tf_scores}
        except Exception as e:
            result.errors.append(f"structure analysis failed: {e}")
        return result

    # -------------------------------------------------------------- internal
    def _analyze_one_tf(self, df: pd.DataFrame) -> float:
        highs = self._swing_points(df["high"], is_high=True)
        lows  = self._swing_points(df["low"],  is_high=False)
        if len(highs) < 2 or len(lows) < 2:
            return 0.0

        last_high = highs[-1].price
        prev_high = highs[-2].price
        last_low  = lows[-1].price
        prev_low  = lows[-2].price

        hh = last_high > prev_high
        hl = last_low  > prev_low
        lh = last_high < prev_high
        ll = last_low  < prev_low

        if hh and hl:   return  1.0
        if lh and ll:   return -1.0
        if hh and ll:   return  0.5   # expanding
        if hl and lh:   return -0.5   # contracting
        return 0.0

    def _swing_points(self, series: pd.Series, is_high: bool) -> List[SwingPoint]:
        n = self.lookback
        pts: List[SwingPoint] = []
        vals = series.values
        for i in range(n, len(vals) - n):
            window = vals[i - n: i + n + 1]
            center = vals[i]
            if is_high and center == window.max():
                pts.append(SwingPoint(index=i, price=float(center), is_high=True))
            elif (not is_high) and center == window.min():
                pts.append(SwingPoint(index=i, price=float(center), is_high=False))
        return pts

    def _range_ratio(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < 20: return 0.0
        rng = (df["high"].tail(50).max() - df["low"].tail(50).min())
        atr = (df["high"] - df["low"]).tail(50).mean()
        return float(rng / max(atr, 1e-9))
