"""
volatility_forecast.py
======================
Tool 158 — EWMA Volatility Forecast (RiskMetrics-style risk gate)

Forecasts next-bar volatility with an exponentially weighted moving
average of squared returns (lambda = 0.94, the RiskMetrics standard)
and compares it with baseline realized volatility.

This tool is a RISK GATE, not a direction caller: when forecast
volatility is expanding far beyond baseline, taking fresh signals is
statistically expensive (slippage, SL hunts, gap risk), so it emits a
neutral score with high confidence — actively pulling the confluence
aggregate toward "stand aside". In calm, contracting volatility it
gently supports the prevailing short-term drift.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult


class VolatilityForecastTool(BaseTool):
    name = "volatility_forecast"

    LAMBDA = 0.94
    LOOKBACK = 150

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 60:
            return result

        closes = df["close"].to_numpy(dtype=float)[-self.LOOKBACK:]
        rets = np.diff(np.log(np.maximum(closes, 1e-12)))
        if len(rets) < 40:
            return result

        # ---- EWMA variance forecast ------------------------------------ #
        lam = self.LAMBDA
        var = float(np.var(rets[:20]))
        for r in rets[20:]:
            var = lam * var + (1 - lam) * r * r
        forecast_vol = float(np.sqrt(max(var, 0)))

        baseline_vol = float(np.std(rets))
        if baseline_vol <= 0:
            return result
        vol_ratio = forecast_vol / baseline_vol      # >1 expanding, <1 calm

        # short-term drift (only supported when vol is calm)
        drift = float(np.tanh(np.mean(rets[-12:]) / (baseline_vol + 1e-12) * 3))

        if vol_ratio > 1.5:
            # volatility shock: stand aside — strong neutral vote
            score = 0.0
            confidence = float(np.clip(0.5 + 0.25 * (vol_ratio - 1.5), 0.5, 0.9))
        elif vol_ratio < 0.85:
            # calm & contracting: support the drift
            score = drift * 0.6
            confidence = 0.55
        else:
            score = drift * 0.3
            confidence = 0.4

        result.score = float(np.clip(score, -1, 1))
        result.confidence = confidence
        result.features = {
            "vol_forecast_ratio": float(np.clip(vol_ratio, 0, 5)),
            "vol_forecast": float(np.clip(forecast_vol / max(closes[-1], 1e-9) * 1e4, 0, 100)),
            "vol_regime_shock": float(vol_ratio > 1.5),
            "calm_drift": drift,
        }
        result.metadata = {
            "forecast_vol": forecast_vol,
            "baseline_vol": baseline_vol,
            "risk_state": "SHOCK" if vol_ratio > 1.5 else
                          ("CALM" if vol_ratio < 0.85 else "NORMAL"),
        }
        return result
