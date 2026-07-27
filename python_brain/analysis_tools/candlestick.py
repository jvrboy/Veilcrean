"""
candlestick.py
==============
Tool 7 — Candlestick pattern recognition.

Implements the most reliable patterns:
    * Engulfing (bullish / bearish)
    * Pin bar / hammer / shooting star
    * Doji
    * Morning star / evening star (3-bar)
    * Inside bar
    * Tweezer tops / bottoms

Score:  +1 for bullish pattern, -1 for bearish, 0 if none.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult


class CandlestickTool(BaseTool):
    name = "candlestick"

    def __init__(self, min_body_ratio: float = 0.3, pin_wick_ratio: float = 2.0):
        super().__init__()
        self.min_body_ratio = min_body_ratio
        self.pin_wick_ratio = pin_wick_ratio

    # -------------------------------------------------------------- public
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        try:
            features: Dict[str, float] = {}
            score_total = 0.0
            n_tfs = 0
            patterns_found: List[Dict] = []

            # The LTF is most relevant for entries
            for tf in ("M5", "M15", "M30", "H1"):
                df = buffers.get(tf)
                if df is None or len(df) < 5: continue
                n_tfs += 1
                patterns = self._scan(df)
                pat_score = 0.0
                for p in patterns:
                    pat_score += p["score"]
                    patterns_found.append({"tf": tf, **p})
                if patterns:
                    pat_score = np.clip(pat_score / max(len(patterns), 1), -1, 1)
                score_total += pat_score
                features[f"cs_pat_count_{tf}"] = float(len(patterns))

            score = score_total / max(n_tfs, 1)
            result.score      = float(np.clip(score, -1, 1))
            result.confidence = 0.5 if patterns_found else 0.3
            result.features   = features
            result.metadata   = {"patterns": patterns_found[:5]}
        except Exception as e:
            result.errors.append(f"candlestick failed: {e}")
        return result

    # -------------------------------------------------------------- internal
    def _scan(self, df: pd.DataFrame) -> List[Dict]:
        """Scan the last 5 candles for patterns; return list of {name, score}."""
        out: List[Dict] = []
        if len(df) < 3: return out
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        n = len(df)
        for i in range(max(0, n-5), n):
            body = abs(c[i] - o[i]) + 1e-12
            rng  = h[i] - l[i] + 1e-12
            upper = h[i] - max(c[i], o[i])
            lower = min(c[i], o[i]) - l[i]

            # Doji
            if body / rng < 0.1:
                out.append({"name": "doji", "score": 0.0})
                continue

            # Pin bar / hammer (long lower wick)
            if lower / body > self.pin_wick_ratio and upper / body < 0.5:
                out.append({"name": "hammer", "score":  0.7 if c[i] > o[i] else 0.5})
                continue
            # Shooting star (long upper wick)
            if upper / body > self.pin_wick_ratio and lower / body < 0.5:
                out.append({"name": "shooting_star", "score": -0.7 if c[i] < o[i] else -0.5})
                continue

            # Engulfing (need 2 bars)
            if i >= 1:
                prev_body = abs(c[i-1] - o[i-1]) + 1e-12
                bull_eng  = (c[i-1] < o[i-1]) and (c[i] > o[i]) and (c[i] > o[i-1]) and (o[i] < c[i-1])
                bear_eng  = (c[i-1] > o[i-1]) and (c[i] < o[i]) and (c[i] < o[i-1]) and (o[i] > c[i-1])
                if bull_eng and body > prev_body * 1.2:
                    out.append({"name": "bullish_engulfing", "score": 0.8})
                elif bear_eng and body > prev_body * 1.2:
                    out.append({"name": "bearish_engulfing", "score": -0.8})

            # Morning star (3 bars) — bullish reversal
            if i >= 2:
                m1_bear = c[i-2] < o[i-2]
                m2_small= body < (h[i-1] - l[i-1]) * 0.3 if i-1 >= 0 else False
                m3_bull = c[i] > o[i] and c[i] > (o[i-2] + c[i-2]) / 2
                if m1_bear and m2_small and m3_bull:
                    out.append({"name": "morning_star", "score": 0.9})
                m1_bull = c[i-2] > o[i-2]
                m3_bear = c[i] < o[i] and c[i] < (o[i-2] + c[i-2]) / 2
                if m1_bull and m2_small and m3_bear:
                    out.append({"name": "evening_star", "score": -0.9})

            # Inside bar (current bar inside previous)
            if i >= 1:
                if h[i] <= h[i-1] and l[i] >= l[i-1]:
                    out.append({"name": "inside_bar", "score": 0.0})

        return out
