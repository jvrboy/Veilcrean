"""
rsi_divergence.py
=================
Tool 156 — RSI Divergence Detector

Detects regular divergences between price swing extremes and RSI:

  * Bullish  — price prints a lower low while RSI prints a higher low
               (selling pressure exhausting → long bias).
  * Bearish  — price prints a higher high while RSI prints a lower high
               (buying pressure exhausting → short bias).

Divergences are one of the highest-probability reversal signals and a
classic way to avoid buying tops / selling bottoms.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = np.full(len(closes), 50.0)
    if len(closes) <= period:
        return rsi
    avg_gain = gains[1:period + 1].mean()
    avg_loss = losses[1:period + 1].mean()
    for i in range(period + 1, len(closes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1 + avg_gain / avg_loss)
    return rsi


def _swings(vals: np.ndarray, width: int = 3) -> Tuple[List[int], List[int]]:
    """Indices of local maxima and minima with a +-width window."""
    highs, lows = [], []
    for i in range(width, len(vals) - width):
        win = vals[i - width:i + width + 1]
        if vals[i] == win.max():
            highs.append(i)
        elif vals[i] == win.min():
            lows.append(i)
    return highs, lows


class RSIDivergenceTool(BaseTool):
    name = "rsi_divergence"

    LOOKBACK = 120
    MAX_AGE = 12          # divergence must involve a recent swing

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 60:
            return result

        closes = df["close"].to_numpy(dtype=float)[-self.LOOKBACK:]
        highs = df["high"].to_numpy(dtype=float)[-self.LOOKBACK:]
        lows = df["low"].to_numpy(dtype=float)[-self.LOOKBACK:]
        rsi = _rsi(closes)
        n = len(closes)

        swing_hi, swing_lo = _swings(closes)

        bull = bear = 0.0
        bull_gap = bear_gap = 0.0

        # bearish: last two swing highs
        if len(swing_hi) >= 2:
            i1, i2 = swing_hi[-2], swing_hi[-1]
            if n - 1 - i2 <= self.MAX_AGE:
                if highs[i2] > highs[i1] and rsi[i2] < rsi[i1] - 1.0:
                    rsi_gap = (rsi[i1] - rsi[i2]) / 30.0
                    bear = float(np.clip(0.5 + rsi_gap, 0, 1))
                    bear_gap = float(rsi[i1] - rsi[i2])

        # bullish: last two swing lows
        if len(swing_lo) >= 2:
            j1, j2 = swing_lo[-2], swing_lo[-1]
            if n - 1 - j2 <= self.MAX_AGE:
                if lows[j2] < lows[j1] and rsi[j2] > rsi[j1] + 1.0:
                    rsi_gap = (rsi[j2] - rsi[j1]) / 30.0
                    bull = float(np.clip(0.5 + rsi_gap, 0, 1))
                    bull_gap = float(rsi[j2] - rsi[j1])

        score = float(np.clip(bull - bear, -1, 1))
        active = max(bull, bear)
        confidence = float(np.clip(0.35 + 0.5 * active, 0.3, 0.85)) if active > 0 else 0.3

        result.score = score
        result.confidence = confidence
        result.features = {
            "divergence_bull": bull,
            "divergence_bear": bear,
            "rsi_now": float(rsi[-1]) / 100.0,
        }
        result.metadata = {
            "bull_rsi_gap": bull_gap,
            "bear_rsi_gap": bear_gap,
            "swing_highs": len(swing_hi),
            "swing_lows": len(swing_lo),
        }
        return result
