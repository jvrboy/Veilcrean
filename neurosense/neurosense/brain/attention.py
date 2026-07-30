"""Attention — salience-driven selection of what deserves processing.

The brain cannot process everything. Attention scores incoming percepts
by novelty (how different from recent input) and intensity, and habituates
to repeated stimuli, exactly like you stop hearing a ticking clock.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class Attention:
    """Novelty + intensity based salience with habituation.

    >>> att = Attention()
    >>> salience = att.evaluate(signature_vector, intensity=0.8)
    >>> if att.is_salient(salience): ...  # worth thinking about
    """

    def __init__(self, history: int = 20, salience_threshold: float = 0.35):
        self.recent: deque[np.ndarray] = deque(maxlen=history)
        self.salience_threshold = salience_threshold
        self._habituation: dict[str, float] = {}

    def evaluate(self, signature: np.ndarray, intensity: float = 0.5,
                 channel: str = "default") -> float:
        """Score 0..1: how much attention does this stimulus deserve?"""
        signature = np.asarray(signature, dtype=np.float64)
        novelty = self._novelty(signature)
        habituation = self._habituation.get(channel, 0.0)
        salience = float(np.clip(
            0.6 * novelty + 0.4 * np.clip(intensity, 0, 1) - habituation,
            0.0, 1.0))
        # Habituate to this channel; recover others slightly
        self._habituation[channel] = min(0.5, habituation + 0.05)
        for ch in self._habituation:
            if ch != channel:
                self._habituation[ch] = max(0.0, self._habituation[ch] - 0.02)
        self.recent.append(signature)
        return salience

    def _novelty(self, signature: np.ndarray) -> float:
        """1.0 = never seen anything like this; 0.0 = identical to recent."""
        if not self.recent:
            return 1.0
        sims = []
        for past in self.recent:
            if past.shape != signature.shape:
                continue
            na, nb = np.linalg.norm(past), np.linalg.norm(signature)
            sims.append(float(past @ signature / (na * nb)) if na and nb else 0.0)
        if not sims:
            return 1.0
        return float(np.clip(1.0 - max(sims), 0.0, 1.0))

    def is_salient(self, salience: float) -> bool:
        return salience >= self.salience_threshold

    def reset_habituation(self) -> None:
        self._habituation.clear()
