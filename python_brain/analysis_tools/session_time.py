"""
session_time.py
===============
Tool 6 — Session & Time Analysis

Outputs a score that reflects *when* we are in the trading day:
    * Asian session   (00:00–08:00 UTC)  — generally low volatility
    * London          (08:00–16:00 UTC)  — high volume, trends
    * New York        (13:00–22:00 UTC)  — high volume
    * London/NY overlap (13:00–16:00 UTC) — peak liquidity
    * Kill zones      — first 1-2 hours of London & NY

It also flags day-of-week patterns (e.g. Wed/Thu often trend best).
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult
from ..config import ANA_CFG


class SessionTimeTool(BaseTool):
    name = "session_time"

    # ------------------------------------------------------------------ API
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        try:
            now: datetime = ctx.get("now") or datetime.now(timezone.utc)

            h = now.hour
            weekday = now.weekday()  # 0 = Mon, 4 = Fri
            in_asian   = ANA_CFG.asian_start  <= h < ANA_CFG.asian_end
            in_london  = ANA_CFG.london_start <= h < ANA_CFG.london_end
            in_ny      = ANA_CFG.ny_start     <= h < ANA_CFG.ny_end
            overlap    = in_london and in_ny

            kill_london = 8  <= h < 10
            kill_ny     = 13 <= h < 15
            is_killzone = kill_london or kill_ny

            # Score: best during overlap, neutral in Asian, positive during London
            if   overlap:    score =  0.8
            elif kill_london or kill_ny: score = 0.7
            elif in_london:  score =  0.5
            elif in_ny:      score =  0.4
            elif in_asian:   score = -0.1
            else:            score =  0.0

            # Day-of-week modifier (Tue-Thu are statistically best)
            dow_score = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.3, 4: 0.1, 5: -0.3, 6: -0.5}.get(weekday, 0.0)
            score = float(np.clip(score + 0.2 * dow_score, -1, 1))

            result.score      = score
            result.confidence = 0.7
            result.features   = {
                "sess_hour":      float(h),
                "sess_weekday":   float(weekday),
                "sess_in_london": float(in_london),
                "sess_in_ny":     float(in_ny),
                "sess_in_asian":  float(in_asian),
                "sess_overlap":   float(overlap),
                "sess_killzone":  float(is_killzone),
            }
            result.metadata = {
                "session":   "overlap" if overlap else
                             "london"   if in_london else
                             "newyork"  if in_ny else
                             "asian"    if in_asian else "other",
                "is_killzone": is_killzone,
            }
        except Exception as e:
            result.errors.append(f"session/time failed: {e}")
        return result
