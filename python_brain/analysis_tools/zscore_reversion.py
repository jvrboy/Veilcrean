"""
zscore_reversion.py
===================
Tool 157 — Z-Score Mean Reversion

Computes how many standard deviations price sits from its rolling mean
and — critically — only fades the extension when the market is actually
mean-reverting (variance-ratio test < 1). Fading extremes in a trending
market is a classic loss generator, so the tool gates itself on the
measured mean-reversion strength of the series.

Score  = +1 deeply oversold in a reverting market … -1 deeply overbought.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult


class ZScoreReversionTool(BaseTool):
    name = "zscore_reversion"

    PERIOD = 40

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < self.PERIOD + 20:
            return result

        closes = df["close"].to_numpy(dtype=float)[-160:]

        window = closes[-self.PERIOD:]
        mean = float(window.mean())
        std = float(window.std())
        if std <= 0:
            return result
        zscore = float((closes[-1] - mean) / std)

        # ---- variance ratio test (q=5): <1 => mean reverting ----------- #
        rets = np.diff(np.log(np.maximum(closes, 1e-12)))
        q = 5
        if len(rets) < q * 6:
            return result
        var1 = float(np.var(rets, ddof=1))
        agg = np.array([rets[i:i + q].sum() for i in range(0, len(rets) - q + 1)])
        varq = float(np.var(agg, ddof=1)) / q
        vr = varq / var1 if var1 > 0 else 1.0

        # reversion strength: 1 when VR is far below 1, 0 when >= 1
        reversion = float(np.clip(1.0 - vr, 0.0, 1.0))

        # only meaningful beyond ~1.5 sigma
        stretch = float(np.clip((abs(zscore) - 1.5) / 1.5, 0.0, 1.0))
        raw = -np.sign(zscore) * stretch          # fade the extension
        score = float(np.clip(raw * (0.25 + 0.75 * reversion), -1, 1))

        confidence = float(np.clip(0.3 + 0.45 * stretch * reversion, 0.25, 0.85))

        result.score = score
        result.confidence = confidence
        result.features = {
            "zscore": float(np.clip(zscore, -4, 4)),
            "variance_ratio": float(np.clip(vr, 0, 3)),
            "reversion_strength": reversion,
            "stretch": stretch,
        }
        result.metadata = {
            "rolling_mean": mean,
            "rolling_std": std,
            "regime": "MEAN_REVERTING" if vr < 0.8 else
                      ("TRENDING" if vr > 1.2 else "NEUTRAL"),
        }
        return result
