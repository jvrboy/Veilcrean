"""
supply_demand.py
================
Tool 2 — Supply & Demand / Order Blocks / Fair Value Gaps

Identifies:
    * Order blocks (last opposing candle before a strong impulse)
    * Fair Value Gaps (3-candle imbalance)
    * Proximity of current price to the nearest S/D zone
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult
from ..config import ANA_CFG, TIMEFRAMES


@dataclass
class Zone:
    tf:       str
    top:      float
    bottom:   float
    is_demand: bool
    strength:  float = 1.0
    age:       int   = 0


class SupplyDemandTool(BaseTool):
    name = "supply_demand"

    def __init__(self, ob_min_pips: float = None, fvg_min_pips: float = None, pip_size: float = 0.0001):
        super().__init__()
        self.ob_min_pips  = ob_min_pips  or ANA_CFG.ob_min_impulse_pips
        self.fvg_min_pips = fvg_min_pips or ANA_CFG.fvg_min_pips
        self.pip_size     = pip_size     # 0.0001 for 5-digit FX, 0.01 for JPY, 0.1 for indices

    # -------------------------------------------------------------- public
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        try:
            pip = ctx.get("pip_size", self.pip_size)
            price = ctx.get("price")
            if price is None:
                # best-effort fallback
                df = buffers.get("M5") or buffers.get("M1") or buffers.get("H1")
                if df is not None and not df.empty:
                    price = float(df["close"].iloc[-1])

            if price is None:
                result.errors.append("no price reference")
                return result

            nearest_demand = None
            nearest_supply = None
            features: Dict[str, float] = {}

            for tf in TIMEFRAMES:
                df = buffers.get(tf)
                if df is None or len(df) < 20: continue

                obs = self._find_order_blocks(df, pip)
                fvgs = self._find_fvgs(df, pip)

                # Add every active zone
                for z in obs + fvgs:
                    dist = self._distance_to_zone(price, z)
                    if z.is_demand and (nearest_demand is None or dist < nearest_demand[0]):
                        nearest_demand = (dist, z)
                    if (not z.is_demand) and (nearest_supply is None or dist < nearest_supply[0]):
                        nearest_supply = (dist, z)

                features[f"sd_ob_count_{tf}"]    = float(len(obs))
                features[f"sd_fvg_count_{tf}"]   = float(len(fvgs))
                features[f"sd_strength_{tf}"]    = self._zone_strength(obs + fvgs)

            score = 0.0
            if nearest_demand and nearest_supply:
                # Closer to demand = bullish; closer to supply = bearish
                d_dem, _ = nearest_demand
                d_sup, _ = nearest_supply
                if d_dem + d_sup > 0:
                    score = float((d_sup - d_dem) / (d_dem + d_sup))
            elif nearest_demand:
                score =  0.4
            elif nearest_supply:
                score = -0.4

            result.score      = float(np.clip(score, -1, 1))
            result.confidence = 0.55
            result.features   = features
            result.metadata   = {
                "nearest_demand_pips": nearest_demand[0]/pip if nearest_demand else None,
                "nearest_supply_pips": nearest_supply[0]/pip   if nearest_supply else None,
            }
        except Exception as e:
            result.errors.append(f"supply/demand failed: {e}")
        return result

    # -------------------------------------------------------------- internal
    def _find_order_blocks(self, df: pd.DataFrame, pip: float) -> List[Zone]:
        zones: List[Zone] = []
        closes = df["close"].values
        opens  = df["open"].values
        highs  = df["high"].values
        lows   = df["low"].values

        min_impulse = self.ob_min_pips * pip

        # Iterate: a bullish OB is the last bearish candle before a strong up-move
        for i in range(2, len(df) - 1):
            impulse = closes[i] - closes[i-1]
            if impulse < min_impulse and closes[i-1] < opens[i-1]:
                zones.append(Zone(
                    tf = df.attrs.get("tf", "?"),
                    top    = highs[i-1],
                    bottom = lows[i-1],
                    is_demand = True,
                    strength = min(impulse / min_impulse, 3.0),
                ))
                if len(zones) > 30: break
            elif -impulse > min_impulse and closes[i-1] > opens[i-1]:
                zones.append(Zone(
                    tf = df.attrs.get("tf", "?"),
                    top    = highs[i-1],
                    bottom = lows[i-1],
                    is_demand = False,
                    strength = min(-impulse / min_impulse, 3.0),
                ))
                if len(zones) > 30: break
        return zones

    def _find_fvgs(self, df: pd.DataFrame, pip: float) -> List[Zone]:
        zones: List[Zone] = []
        min_gap = self.fvg_min_pips * pip
        highs = df["high"].values
        lows  = df["low"].values
        opens = df["open"].values
        closes= df["close"].values
        for i in range(2, len(df)):
            # Bullish FVG: high[i-2] < low[i]
            if lows[i] - highs[i-2] > min_gap and closes[i] > opens[i]:
                zones.append(Zone(
                    tf = df.attrs.get("tf", "?"),
                    top    = lows[i],
                    bottom = highs[i-2],
                    is_demand = True,
                    strength  = (lows[i] - highs[i-2]) / min_gap,
                ))
            # Bearish FVG: high[i] < low[i-2]
            elif highs[i-2] - lows[i] > min_gap and closes[i] < opens[i]:
                zones.append(Zone(
                    tf = df.attrs.get("tf", "?"),
                    top    = lows[i-2],
                    bottom = highs[i],
                    is_demand = False,
                    strength  = (highs[i-2] - lows[i]) / min_gap,
                ))
            if len(zones) > 30: break
        return zones

    @staticmethod
    def _distance_to_zone(price: float, zone: Zone) -> float:
        if zone.bottom <= price <= zone.top: return 0.0
        return min(abs(price - zone.top), abs(price - zone.bottom))

    @staticmethod
    def _zone_strength(zones: List[Zone]) -> float:
        if not zones: return 0.0
        return float(np.clip(np.mean([z.strength for z in zones]) / 3.0, 0.0, 1.0))
