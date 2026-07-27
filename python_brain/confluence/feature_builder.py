"""
feature_builder.py
==================
Combines the outputs of the 8 analysis tools into a single, fixed-size
numeric feature vector suitable for the neural network.

The vector is ordered, normalized, and **deterministic** — the NN
expects a specific column order, and the order is saved in
`self.feature_names` so the same names can be used at inference time.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np

from ..analysis_tools.base_tool import ToolResult


class FeatureBuilder:
    """Flattens tool results into a numeric vector."""

    def __init__(self):
        self.feature_names: List[str] = []
        self._built = False

    # ------------------------------------------------------------------ API
    def build(self, tool_results: Dict[str, ToolResult], context: Optional[dict] = None) -> np.ndarray:
        """Return a 1D float32 array of all features in a fixed order."""
        ctx = context or {}
        features: Dict[str, float] = {}

        # 1. Top-level per-tool scores and confidences
        for name, res in tool_results.items():
            features[f"{name}__score"]      = float(np.clip(res.score, -1, 1))
            features[f"{name}__confidence"] = float(np.clip(res.confidence, 0, 1))

        # 2. Per-tool inner features (RSI/MACD/slopes/...)
        for name, res in tool_results.items():
            for k, v in (res.features or {}).items():
                fk = f"{name}__{k}"
                if fk not in features:
                    try:
                        features[fk] = float(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0))
                    except Exception:
                        features[fk] = 0.0

        # 3. Context features (price vs. moving averages, spread, vol, regime placeholders)
        price = ctx.get("price")
        if price is not None and price > 0:
            features["ctx__price_norm"] = float(price / 1.0)  # raw — model can learn
        features["ctx__spread_pts"]  = float(ctx.get("spread", 0.0))
        features["ctx__hour"]        = float(ctx.get("hour", 0))
        features["ctx__weekday"]     = float(ctx.get("weekday", 0))
        features["ctx__open_pos"]    = float(ctx.get("open_positions", 0))
        features["ctx__account_dd"]  = float(ctx.get("account_dd_pct", 0.0))
        features["ctx__regime_trending"]  = float(ctx.get("regime_trending", 0))
        features["ctx__regime_ranging"]   = float(ctx.get("regime_ranging", 0))
        features["ctx__regime_volatile"]  = float(ctx.get("regime_volatile", 0))
        features["ctx__regime_choppy"]    = float(ctx.get("regime_choppy", 0))
        features["ctx__regime_breakout"]  = float(ctx.get("regime_breakout", 0))

        # 4. Build the ordered array
        if not self._built:
            self.feature_names = sorted(features.keys())
            self._built = True
        # If a new feature slipped in, expand the names list
        for k in features.keys():
            if k not in self.feature_names:
                self.feature_names.append(k)

        vec = np.array([features.get(name, 0.0) for name in self.feature_names],
                       dtype=np.float32)
        return vec

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    def names(self) -> List[str]:
        return list(self.feature_names)
