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
from typing import Dict, Optional, Union

from ..config import TIMEFRAMES, CANDLE_HISTORY
from .data_cleaner  import DataCleaner
from .normalizer    import Normalizer


class SafeDataFrame(pd.DataFrame):
    """DataFrame with a defined truthiness.

    Many analysis tools use the ``buffers.get("M15") or buffers.get("H1")``
    fallback pattern. A plain ``pd.DataFrame`` raises
    ``ValueError: The truth value of a DataFrame is ambiguous`` when tested
    in boolean context, which silently disabled every such tool. This
    subclass defines ``__bool__`` as "has rows", so those fallbacks work.
    """

    @property
    def _constructor(self):
        return SafeDataFrame

    def __bool__(self) -> bool:
        return len(self.index) > 0


def safe_frame(df: Optional[pd.DataFrame]):
    """Wrap a DataFrame in :class:`SafeDataFrame` (idempotent, None-safe).

    Non-DataFrame values (lists, dicts, scalars) are returned unchanged so
    callers that pass non-tabular buffer values keep working.
    """
    if df is None:
        return SafeDataFrame()
    if isinstance(df, SafeDataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return SafeDataFrame(df)
    return df


class BufferManager:
    """Owns the rolling candle history for the bot."""

    def __init__(self, max_len: int = CANDLE_HISTORY):
        self.max_len = max_len
        self.cleaner   = DataCleaner()
        self.normalizer = Normalizer()
        self._buffers: Dict[str, pd.DataFrame] = {tf: SafeDataFrame() for tf in TIMEFRAMES}

    # ------------------------------------------------------------------ API
    def update(self, candles: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Merge incoming candles into the buffer, clean, return."""
        if not candles:
            return self._buffers

        for tf, df in candles.items():
            if df is None or df.empty:
                continue
            existing = self._buffers.get(tf, SafeDataFrame())
            merged   = self._merge(existing, df)
            merged   = self.cleaner.clean(merged)
            merged   = self.normalizer.fit_transform(merged, tf)
            merged   = merged.tail(self.max_len)
            self._buffers[tf] = safe_frame(merged)
        return self._buffers

    def get(self, tf: str) -> pd.DataFrame:
        return safe_frame(self._buffers.get(tf, SafeDataFrame()))

    def all(self) -> Dict[str, pd.DataFrame]:
        return {tf: safe_frame(df) for tf, df in self._buffers.items()}

    def reset(self) -> None:
        self._buffers = {tf: SafeDataFrame() for tf in TIMEFRAMES}

    # ------------------------------------------------------------------ internal
    def _merge(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if existing is None or existing.empty:
            return new
        # Use last 100 rows of existing as a tail to be safe with partial overlap
        combined = pd.concat([existing.tail(100), new], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        return combined

