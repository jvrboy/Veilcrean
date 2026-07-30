"""
kalman_trend.py
===============
Tool 154 — Kalman Filter Trend Estimator

Runs a constant-velocity Kalman filter over closing prices to extract a
noise-free trend estimate and its velocity. Unlike moving averages, the
Kalman filter adapts its lag automatically to how noisy the series is,
so trend turns are picked up earlier with fewer whipsaws.

Score  = tanh of ATR-normalized filtered velocity (+ = up-trend).
Confidence rises when the raw price hugs the filtered estimate
(low innovation), i.e. when the trend read is trustworthy.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult


class KalmanTrendTool(BaseTool):
    name = "kalman_trend"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 40:
            return result

        closes = df["close"].to_numpy(dtype=float)[-200:]
        n = len(closes)

        # ---- constant-velocity Kalman filter -------------------------- #
        # state: [level, velocity]
        dt = 1.0
        F = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        # measurement noise from recent return variance
        rets = np.diff(closes)
        r_var = max(float(np.var(rets)), 1e-12)
        R = np.array([[r_var * 4.0]])
        q = r_var * 0.05
        Q = np.array([[q * dt ** 3 / 3, q * dt ** 2 / 2],
                      [q * dt ** 2 / 2, q * dt]])

        x = np.array([closes[0], 0.0])
        P = np.eye(2) * r_var
        innovations = []
        for z in closes[1:]:
            # predict
            x = F @ x
            P = F @ P @ F.T + Q
            # update
            y = z - (H @ x)[0]
            S = (H @ P @ H.T)[0, 0] + R[0, 0]
            K = (P @ H.T).flatten() / S
            x = x + K * y
            P = (np.eye(2) - np.outer(K, H.flatten())) @ P
            innovations.append(abs(y))

        level, velocity = float(x[0]), float(x[1])

        # ---- normalize ------------------------------------------------- #
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            atr = max(float(np.std(rets)), 1e-9)

        norm_vel = velocity / atr
        score = float(np.tanh(2.5 * norm_vel))

        # confidence: small recent innovation relative to ATR = clean trend
        recent_innov = float(np.mean(innovations[-10:])) if innovations else atr
        innov_ratio = recent_innov / atr
        confidence = float(np.clip(0.85 - 0.5 * innov_ratio, 0.2, 0.9))

        result.score = score
        result.confidence = confidence
        result.features = {
            "kalman_velocity_atr": float(np.clip(norm_vel, -5, 5)),
            "kalman_gap_atr": float(np.clip((closes[-1] - level) / atr, -5, 5)),
            "kalman_innovation_ratio": float(np.clip(innov_ratio, 0, 5)),
        }
        result.metadata = {
            "level": level,
            "velocity": velocity,
            "atr": atr,
        }
        return result
