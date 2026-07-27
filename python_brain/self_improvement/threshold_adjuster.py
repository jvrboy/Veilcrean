"""
threshold_adjuster.py
====================
Dynamically raises / lowers the confidence threshold based on recent
performance. Goal: take more trades in good regimes, fewer in bad ones.
"""
from __future__ import annotations
from collections import deque
from typing import Deque

from ..config import SI_CFG


class ThresholdAdjuster:
    def __init__(self, window: int = 30):
        self.window = window
        self.recent: Deque[float] = deque(maxlen=window)   # 1 = win, 0 = loss
        self.threshold = SI_CFG.confidence_threshold

    def record_outcome(self, is_win: bool) -> None:
        self.recent.append(1.0 if is_win else 0.0)

    def update(self) -> float:
        """Recompute and return the threshold for the next decision."""
        if len(self.recent) < 10:
            return self.threshold
        wr = sum(self.recent) / len(self.recent)
        if wr < 0.40:
            self.threshold = min(SI_CFG.confidence_threshold_max, self.threshold + 0.05)
        elif wr > 0.60:
            self.threshold = max(0.50, self.threshold - 0.02)
        return self.threshold

    @property
    def current(self) -> float:
        return self.threshold
