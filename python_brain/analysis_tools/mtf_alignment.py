"""
mtf_alignment.py
================
Tool 8 — Multi-Timeframe Alignment

Computes how aligned the timeframes are. We use a simple trend proxy
(EMA-50 slope) on each TF and report the percentage of TFs that agree
on direction.
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult
from ..config import TIMEFRAMES


class MTFAlignmentTool(BaseTool):
    name = "mtf_alignment"

    def __init__(self, ema_period: int = 50):
        super().__init__()
        self.ema_period = ema_period

    # -------------------------------------------------------------- public
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        try:
            trends: Dict[str, int] = {}
            features: Dict[str, float] = {}

            for tf in TIMEFRAMES:
                df = buffers.get(tf)
                if df is None or len(df) < self.ema_period + 5: continue
                ema = df["close"].ewm(span=self.ema_period, adjust=False).mean()
                slope = float(ema.iloc[-1] - ema.iloc[-5])
                trends[tf] = int(np.sign(slope))
                features[f"mtf_slope_{tf}"] = slope

            if not trends:
                result.errors.append("no TF data")
                return result

            vals = list(trends.values())
            up   = sum(1 for v in vals if v > 0)
            down = sum(1 for v in vals if v < 0)
            total = len(vals)
            net = (up - down) / total                # -1..+1

            # HTF bias (H4/D1) — counted double
            htf_bias = 0
            for tf in ("H4", "D1", "W1"):
                if tf in trends:
                    htf_bias += trends[tf]
            htf_norm = float(np.clip(htf_bias / 3.0, -1, 1))

            result.score      = float(np.clip(0.6 * net + 0.4 * htf_norm, -1, 1))
            result.confidence = max(0.3, abs(net))
            result.features   = features
            result.metadata   = {"trends": trends, "up_pct": up/total, "down_pct": down/total}
        except Exception as e:
            result.errors.append(f"mtf alignment failed: {e}")
        return result
