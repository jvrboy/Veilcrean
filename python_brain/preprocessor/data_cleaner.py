"""
data_cleaner.py
===============
Cleans raw OHLCV DataFrames coming from the EA.

Responsibilities
----------------
* Forward-fill small gaps in candle history
* Drop candles with zero / negative width
* Detect and cap obvious data glitches (price jumps > 5σ)
* Add a `returns` column (close-to-close log returns)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from typing import Dict


class DataCleaner:
    """Stateless cleaner. Apply on every buffer update."""

    # ------------------------------------------------------------------ API
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean a single TF DataFrame in-place-style. Returns a copy."""
        if df is None or df.empty:
            return df

        out = df.copy()

        # 1. Sort by time
        if not out.index.is_monotonic_increasing:
            out = out.sort_index()

        # 2. Drop duplicates
        out = out[~out.index.duplicated(keep="last")]

        # 3. Forward-fill small gaps (max 3 periods)
        out = out.ffill(limit=3)

        # 4. Remove zero-width or impossible candles
        valid = (out["high"] >= out["low"]) & (out["open"] > 0) & (out["close"] > 0)
        out = out[valid]

        # 5. Cap spike glitches (> 5σ close-to-close)
        if len(out) > 30:
            rets = out["close"].pct_change()
            std  = rets.std()
            mean = rets.mean()
            if std and not np.isnan(std):
                spike = (rets - mean).abs() > 5 * std
                if spike.any():
                    out.loc[spike, "close"] = out["close"].shift(1)[spike]

        # 6. Add log returns column (NaN on first row)
        out["log_return"] = np.log(out["close"] / out["close"].shift(1))

        return out

    def clean_all(self, candles: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean all TF DataFrames in one call."""
        return {tf: self.clean(df) for tf, df in candles.items() if df is not None}
