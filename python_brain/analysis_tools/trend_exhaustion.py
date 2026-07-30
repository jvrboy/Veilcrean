"""
trend_exhaustion.py
===================
Tool 160 — Trend Exhaustion / Climax Detector

Chasing an old, over-extended move is one of the biggest sources of
retail losses. This tool measures how "late" the current trend is:

  * Run length      — consecutive closes in one direction (TD-style).
  * Extension       — distance from the 50-bar mean in ATR units.
  * Climax bar      — range expansion + close far from the extreme
                      (buying/selling climax signature).

When exhaustion is high, the tool votes AGAINST the trend direction —
warning the confluence engine off late entries and flagging potential
reversals.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult


class TrendExhaustionTool(BaseTool):
    name = "trend_exhaustion"

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 60:
            return result

        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        opens = df["open"].to_numpy(dtype=float) if "open" in df.columns else closes

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            return result

        # ---- run length (consecutive same-direction closes) ------------ #
        diffs = np.sign(np.diff(closes[-15:]))
        run_dir = diffs[-1] if diffs[-1] != 0 else 0.0
        run_len = 0
        for d in diffs[::-1]:
            if d == run_dir and d != 0:
                run_len += 1
            else:
                break
        run_score = float(np.clip((run_len - 4) / 5.0, 0, 1))   # 4+ closes = tiring

        # ---- extension from 50-bar mean --------------------------------- #
        mean50 = float(np.mean(closes[-50:]))
        ext_atr = (closes[-1] - mean50) / atr
        ext_dir = float(np.sign(ext_atr))
        ext_score = float(np.clip((abs(ext_atr) - 2.0) / 3.0, 0, 1))

        # ---- climax bar -------------------------------------------------- #
        last_range = highs[-1] - lows[-1]
        avg_range = float(np.mean(highs[-20:] - lows[-20:]))
        range_exp = last_range / avg_range if avg_range > 0 else 1.0
        body_pos = 0.5
        if last_range > 0:
            body_pos = (closes[-1] - lows[-1]) / last_range   # 1 = closed at high
        climax = 0.0
        if range_exp > 1.8:
            if ext_dir > 0 and body_pos < 0.4:      # up-move, big bar, weak close
                climax = float(np.clip((range_exp - 1.8) / 1.5 + (0.4 - body_pos), 0, 1))
            elif ext_dir < 0 and body_pos > 0.6:    # down-move, big bar, strong close
                climax = float(np.clip((range_exp - 1.8) / 1.5 + (body_pos - 0.6), 0, 1))

        # trend direction to vote against
        trend_dir = ext_dir if ext_dir != 0 else run_dir
        exhaustion = float(np.clip(0.4 * run_score + 0.35 * ext_score + 0.25 * climax, 0, 1))

        # vote against the tired trend, scaled by exhaustion level
        score = float(np.clip(-trend_dir * exhaustion, -1, 1))
        confidence = float(np.clip(0.3 + 0.5 * exhaustion, 0.3, 0.8))

        result.score = score
        result.confidence = confidence
        result.features = {
            "exhaustion_level": exhaustion,
            "run_length": float(run_len),
            "extension_atr": float(np.clip(ext_atr, -8, 8)),
            "climax_signal": climax,
            "range_expansion": float(np.clip(range_exp, 0, 5)),
        }
        result.metadata = {
            "trend_dir": trend_dir,
            "body_pos": float(body_pos),
        }
        return result
