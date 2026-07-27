"""
liquidity.py
============
Tool 3 — Liquidity analysis

Detects:
    * Equal highs / equal lows   (liquidity pools)
    * Stop hunts / liquidity sweeps
    * Liquidity voids (large displacement candles)

Output
------
    score:    +1 = buy-side liquidity likely to be taken (sell),
              -1 = sell-side liquidity likely to be taken (buy)
    sweep_detected: boolean flag
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult
from ..config import ANA_CFG, TIMEFRAMES


class LiquidityTool(BaseTool):
    name = "liquidity"

    def __init__(self, eq_tolerance_pips: float = None, pip_size: float = 0.0001):
        super().__init__()
        self.eq_tol_pips = eq_tolerance_pips or ANA_CFG.eqh_eql_tolerance_pips
        self.pip_size    = pip_size

    # -------------------------------------------------------------- public
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        try:
            pip   = ctx.get("pip_size", self.pip_size)
            price = ctx.get("price")
            if price is None:
                df = buffers.get("M5") or buffers.get("H1")
                if df is not None and not df.empty:
                    price = float(df["close"].iloc[-1])

            score = 0.0
            sweep_buy_side  = False
            sweep_sell_side = False
            features: Dict[str, float] = {}

            for tf in TIMEFRAMES:
                df = buffers.get(tf)
                if df is None or len(df) < 30: continue

                tol = self.eq_tol_pips * pip
                eqh = self._equal_highs(df, tol)
                eql = self._equal_lows(df,  tol)
                buy_sweep, sell_sweep = self._sweeps(df, eqh, eql, tol)

                features[f"liq_eqh_{tf}"]     = float(len(eqh))
                features[f"liq_eql_{tf}"]     = float(len(eql))
                features[f"liq_sweep_buy_{tf}"]  = 1.0 if buy_sweep  else 0.0
                features[f"liq_sweep_sell_{tf}"] = 1.0 if sell_sweep else 0.0

                if buy_sweep:  sweep_buy_side  = True
                if sell_sweep: sweep_sell_side = True

            # Sweep of sell-side liquidity (lows taken then price reverses up) = bullish
            # Sweep of buy-side liquidity (highs taken then price reverses down) = bearish
            if sweep_sell_side and not sweep_buy_side: score =  0.8
            elif sweep_buy_side and not sweep_sell_side: score = -0.8
            elif sweep_sell_side and sweep_buy_side:    score =  0.0
            else: score = 0.0

            result.score      = float(np.clip(score, -1, 1))
            result.confidence = 0.6 if (sweep_buy_side or sweep_sell_side) else 0.4
            result.features   = features
            result.metadata   = {
                "sweep_buy_side":  sweep_buy_side,
                "sweep_sell_side": sweep_sell_side,
            }
        except Exception as e:
            result.errors.append(f"liquidity analysis failed: {e}")
        return result

    # -------------------------------------------------------------- internal
    def _equal_highs(self, df: pd.DataFrame, tol: float) -> List[float]:
        highs = df["high"].values
        eq: List[float] = []
        last_pick = -10
        for i in range(2, len(highs) - 2):
            if highs[i] == highs[i-2:i+3].max() and i - last_pick > 5:
                if eq and abs(highs[i] - eq[-1]) < tol:
                    eq[-1] = (eq[-1] + highs[i]) / 2.0
                else:
                    eq.append(float(highs[i]))
                last_pick = i
        return eq[-5:]

    def _equal_lows(self, df: pd.DataFrame, tol: float) -> List[float]:
        lows = df["low"].values
        eq: List[float] = []
        last_pick = -10
        for i in range(2, len(lows) - 2):
            if lows[i] == lows[i-2:i+3].min() and i - last_pick > 5:
                if eq and abs(lows[i] - eq[-1]) < tol:
                    eq[-1] = (eq[-1] + lows[i]) / 2.0
                else:
                    eq.append(float(lows[i]))
                last_pick = i
        return eq[-5:]

    def _sweeps(self, df: pd.DataFrame, eqh: List[float], eql: List[float], tol: float):
        """A sweep is when price *takes out* an equal-high/low then closes
        back inside."""
        buy_sweep  = False
        sell_sweep = False
        if len(df) < 4: return buy_sweep, sell_sweep
        last_high = float(df["high"].iloc[-1])
        last_low  = float(df["low"].iloc[-1])
        last_close= float(df["close"].iloc[-1])
        prev_close= float(df["close"].iloc[-2])

        for h in eqh:
            if last_high > h + tol and last_close < h and prev_close < h:
                buy_sweep = True
                break
        for l in eql:
            if last_low < l - tol and last_close > l and prev_close > l:
                sell_sweep = True
                break
        return buy_sweep, sell_sweep
