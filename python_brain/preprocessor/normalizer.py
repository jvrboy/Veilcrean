"""
normalizer.py
=============
Feature normalization for the neural networks.

We use a *rolling* z-score scaler so the network is never fed values
from a different volatility regime than what it's about to predict on.
The scaler is fit on a per-timeframe basis, but at inference time we
re-use the most-recent fit to avoid leakage.

Two modes
---------
* `fit=True`  — fit on the supplied data (used during training)
* `fit=False` — apply previously-fitted scaling (used at inference)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class Normalizer:
    """Per-TF, per-column rolling z-score scaler."""

    def __init__(self, window: int = 500, clip: float = 5.0):
        self.window = window
        self.clip   = clip
        self._stats: Dict[str, Dict[str, Tuple[float, float]]] = {}

    # ------------------------------------------------------------------ API
    def fit(self, df: pd.DataFrame, tf: str) -> None:
        """Fit a scaler on the tail of `df` (length = `window`)."""
        if df is None or df.empty:
            return
        tail = df.tail(self.window)
        stats: Dict[str, Tuple[float, float]] = {}
        for col in tail.select_dtypes(include="number").columns:
            mu = float(tail[col].mean())
            sd = float(tail[col].std(ddof=0))
            if sd == 0 or np.isnan(sd):
                sd = 1.0
            stats[col] = (mu, sd)
        self._stats[tf] = stats

    def transform(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        if df is None or df.empty or tf not in self._stats:
            return df
        out = df.copy()
        for col, (mu, sd) in self._stats[tf].items():
            if col in out.columns:
                out[col] = ((out[col] - mu) / sd).clip(-self.clip, self.clip)
        return out

    def fit_transform(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        self.fit(df, tf)
        return self.transform(df, tf)

    # ------------------------------------------------------------------ helpers
    def state_dict(self) -> dict:
        return {"window": self.window, "clip": self.clip, "stats": self._stats}

    def load_state_dict(self, state: dict) -> None:
        self.window = state.get("window", self.window)
        self.clip   = state.get("clip",   self.clip)
        self._stats = state.get("stats",  {})
