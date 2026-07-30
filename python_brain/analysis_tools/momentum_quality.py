"""
momentum_quality.py
===================
Tool 159 — Momentum Quality Index

Raw momentum tells you price moved; momentum QUALITY tells you whether
that move is tradable. This tool grades the recent move on three axes:

  1. Consistency — fraction of bars agreeing with the move's direction.
  2. Smoothness  — R-squared of a linear fit over the move (path
                   efficiency: straight lines continue, jagged ones
                   revert).
  3. Volume support — is volume expanding with the move?

Only high-quality momentum earns a strong score. Erratic momentum —
the kind that produces breakout traps and immediate reversals — is
scored near zero regardless of magnitude, filtering a major loss
source.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult


class MomentumQualityTool(BaseTool):
    name = "momentum_quality"

    WINDOW = 20

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < self.WINDOW + 15:
            return result

        w = self.WINDOW
        closes = df["close"].to_numpy(dtype=float)
        window = closes[-w:]
        rets = np.diff(window)
        if len(rets) == 0 or np.all(rets == 0):
            return result

        direction = float(np.sign(window[-1] - window[0]))
        if direction == 0:
            return result

        # 1. consistency: share of bars moving with the overall direction
        agree = float(np.mean(np.sign(rets) == direction))

        # 2. smoothness: R^2 of linear fit
        x = np.arange(w, dtype=float)
        slope, intercept = np.polyfit(x, window, 1)
        fitted = slope * x + intercept
        ss_res = float(np.sum((window - fitted) ** 2))
        ss_tot = float(np.sum((window - window.mean()) ** 2))
        r2 = float(np.clip(1 - ss_res / ss_tot, 0, 1)) if ss_tot > 0 else 0.0

        # 3. volume support
        vol_support = 0.5
        if "volume" in df.columns:
            vols = df["volume"].to_numpy(dtype=float)
            if len(vols) >= 2 * w and np.nansum(vols[-2 * w:]) > 0:
                recent = float(np.nanmean(vols[-w:]))
                prior = float(np.nanmean(vols[-2 * w:-w]))
                if prior > 0:
                    vol_support = float(np.clip(recent / prior / 2.0, 0, 1))

        # magnitude in ATR units
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            atr = max(float(np.std(rets)), 1e-9)
        magnitude = float(np.clip(abs(window[-1] - window[0]) / (atr * 4), 0, 1))

        quality = float(np.clip(0.5 * agree + 0.35 * r2 + 0.15 * vol_support, 0, 1))
        score = float(np.clip(direction * magnitude * quality * 1.6, -1, 1))
        confidence = float(np.clip(0.25 + 0.6 * quality, 0.25, 0.85))

        result.score = score
        result.confidence = confidence
        result.features = {
            "momo_consistency": agree,
            "momo_smoothness_r2": r2,
            "momo_volume_support": vol_support,
            "momo_quality": quality,
            "momo_magnitude_atr": magnitude,
        }
        result.metadata = {
            "direction": direction,
            "slope": float(slope),
        }
        return result
