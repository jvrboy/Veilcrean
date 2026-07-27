"""
buffer_manager.py
=================
Rolling per-timeframe candle buffers.

On every market-data packet from the EA we update the buffer for each
timeframe. Old candles drop off the tail; new ones are appended.
The buffer is the single source of truth for what the analysis tools
look at.
"""
from __future__ import annotations
import pandas as pd
from typing import Dict, Optional

from ..config import TIMEFRAMES, CANDLE_HISTORY
from .data_cleaner  import DataCleaner
from .normalizer    import Normalizer


class BufferManager:
    """Owns the rolling candle history for the bot."""

    def __init__(self, max_len: int = CANDLE_HISTORY):
        self.max_len = max_len
        self.cleaner   = DataCleaner()
        self.normalizer = Normalizer()
        self._buffers: Dict[str, pd.DataFrame] = {tf: pd.DataFrame() for tf in TIMEFRAMES}

    # ------------------------------------------------------------------ API
    def update(self, candles: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Merge incoming candles into the buffer, clean, return."""
        if not candles:
            return self._buffers

        for tf, df in candles.items():
            if df is None or df.empty:
                continue
            existing = self._buffers.get(tf, pd.DataFrame())
            merged   = self._merge(existing, df)
            merged   = self.cleaner.clean(merged)
            merged   = self.normalizer.fit_transform(merged, tf)
            merged   = merged.tail(self.max_len)
            self._buffers[tf] = merged
        return self._buffers

    def get(self, tf: str) -> pd.DataFrame:
        return self._buffers.get(tf, pd.DataFrame())

    def all(self) -> Dict[str, pd.DataFrame]:
        return dict(self._buffers)

    def reset(self) -> None:
        self._buffers = {tf: pd.DataFrame() for tf in TIMEFRAMES}

    # ------------------------------------------------------------------ internal
    def _merge(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if existing is None or existing.empty:
            return new
        # Use last 100 rows of existing as a tail to be safe with partial overlap
        combined = pd.concat([existing.tail(100), new], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        return combined
