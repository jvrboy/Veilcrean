"""Attention mechanisms for numeric feature vectors and signal dictionaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def soft_attention(values: np.ndarray, query: np.ndarray | None = None, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted sum and attention weights for a value matrix."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if query is None:
        query = values.mean(axis=0)
    query = np.asarray(query, dtype=float)
    if query.shape[-1] != values.shape[-1]:
        raise ValueError("query and values must share the feature dimension")
    scores = values @ query / max(temperature, 1e-9)
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / np.sum(weights)
    return weights @ values, weights


def top_k(scores: Sequence[float], k: int = 1) -> list[int]:
    """Return indices of the top ``k`` scores descending."""
    if k <= 0:
        return []
    arr = np.asarray(scores, dtype=float)
    return list(np.argsort(arr)[::-1][:k].astype(int))


@dataclass(frozen=True)
class FocusState:
    indices: list[int]
    weights: list[float]
    salience: float


@dataclass
class AttentionMechanism:
    """Compute salience from feature dictionaries or numeric vectors."""

    capacity: int = 3
    temperature: float = 1.0

    def score(self, item) -> float:
        if isinstance(item, dict):
            if "features" in item and isinstance(item["features"], dict):
                values = item["features"].values()
            else:
                values = item.values()
            nums = [float(v) for v in values if isinstance(v, (int, float, bool, np.number))]
            return float(np.linalg.norm(nums)) if nums else 0.0
        arr = np.asarray(item, dtype=float)
        return float(np.linalg.norm(arr))

    def focus(self, items: Iterable) -> FocusState:
        items = list(items)
        scores = np.array([self.score(item) for item in items], dtype=float)
        if len(scores) == 0:
            return FocusState([], [], 0.0)
        selected = top_k(scores, min(self.capacity, len(scores)))
        sel_scores = scores[selected] / max(self.temperature, 1e-9)
        sel_scores = sel_scores - np.max(sel_scores)
        weights = np.exp(sel_scores)
        weights = weights / weights.sum()
        return FocusState(indices=selected, weights=[float(w) for w in weights], salience=float(scores[selected].mean()))

    __call__ = focus
