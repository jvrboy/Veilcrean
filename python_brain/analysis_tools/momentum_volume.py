"""
momentum_volume.py
==================
Tool 4 — Momentum & Volume

Computes (per timeframe):
    * RSI(14)
    * MACD(12, 26, 9)
    * Volume z-score (relative to last 50 bars)
    * Bullish / bearish divergence flag (price vs RSI)
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult
from ..config import ANA_CFG, TIMEFRAMES


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = -delta.clip(upper=0).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast,   adjust=False).mean()
    ema_slow = series.ewm(span=slow,   adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal,  adjust=False).mean()
    hist     = macd - sig
    return macd, sig, hist


class MomentumVolumeTool(BaseTool):
    name = "momentum_volume"

    def __init__(self,
                 rsi_period: int = None,
                 macd_fast:  int = None,
                 macd_slow:  int = None,
                 macd_sig:   int = None):
        super().__init__()
        self.rsi_period = rsi_period or ANA_CFG.rsi_period
        self.macd_fast  = macd_fast  or ANA_CFG.macd_fast
        self.macd_slow  = macd_slow  or ANA_CFG.macd_slow
        self.macd_sig   = macd_sig   or ANA_CFG.macd_signal

    # -------------------------------------------------------------- public
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        try:
            score_total  = 0.0
            div_bull     = False
            div_bear     = False
            features: Dict[str, float] = {}
            n_tfs = 0

            for tf in TIMEFRAMES:
                df = buffers.get(tf)
                if df is None or len(df) < 50: continue
                n_tfs += 1

                rsi = _rsi(df["close"], self.rsi_period)
                macd, sig, hist = _macd(df["close"], self.macd_fast, self.macd_slow, self.macd_sig)
                vol = df["volume"] if "volume" in df.columns else (df["high"] - df["low"])
                vol_z = (vol - vol.rolling(50).mean()) / (vol.rolling(50).std() + 1e-9)

                rsi_v   = float(rsi.iloc[-1])
                macd_v  = float(macd.iloc[-1])
                sig_v   = float(sig.iloc[-1])
                hist_v  = float(hist.iloc[-1])
                vol_v   = float(vol_z.iloc[-1])

                # Normalize each to [-1, 1]
                rsi_n   = (rsi_v - 50.0) / 50.0
                macd_n  = np.tanh(macd_v * 1000)    # tanh squash
                hist_n  = np.tanh(hist_v * 1000)
                vol_n   = np.tanh(vol_v)

                tf_score = 0.5 * rsi_n + 0.3 * hist_n + 0.2 * vol_n
                score_total += np.clip(tf_score, -1, 1)

                # Divergence: price makes lower low but RSI makes higher low (bull)
                if len(df) >= 30:
                    p_now, p_prev = df["close"].iloc[-1], df["close"].iloc[-30]
                    r_now, r_prev = rsi.iloc[-1],     rsi.iloc[-30]
                    if p_now < p_prev and r_now > r_prev: div_bull = True
                    if p_now > p_prev and r_now < r_prev: div_bear = True

                features[f"mom_rsi_{tf}"]      = rsi_n
                features[f"mom_macd_{tf}"]     = macd_n
                features[f"mom_hist_{tf}"]     = hist_n
                features[f"mom_volz_{tf}"]     = vol_n
                features[f"mom_rsi_raw_{tf}"]  = rsi_v

            score = score_total / max(n_tfs, 1)
            if div_bull: score = float(np.clip(score + 0.3, -1, 1))
            if div_bear: score = float(np.clip(score - 0.3, -1, 1))

            result.score      = float(np.clip(score, -1, 1))
            result.confidence = 0.55
            result.features   = features
            result.metadata   = {"div_bull": div_bull, "div_bear": div_bear}
        except Exception as e:
            result.errors.append(f"momentum/volume failed: {e}")
        return result
