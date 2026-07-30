"""Associative memory store."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Iterable, List

import numpy as np


def _embed(content: Any, dims: int = 64) -> np.ndarray:
    if isinstance(content, np.ndarray):
        arr = content.astype(float).reshape(-1)
        if arr.size >= dims:
            return arr[:dims]
        return np.pad(arr, (0, dims - arr.size))
    if isinstance(content, (list, tuple)) and content and all(isinstance(x, (int, float, bool)) for x in content):
        arr = np.asarray(content, dtype=float).reshape(-1)
        if arr.size >= dims:
            return arr[:dims]
        return np.pad(arr, (0, dims - arr.size))
    text = str(content).lower()
    vec = np.zeros(dims, dtype=float)
    for token in text.split():
        vec[hash(token) % dims] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 1e-12 else float(np.dot(a, b) / denom)


@dataclass
class MemoryItem:
    content: Any
    tags: set[str] = field(default_factory=set)
    strength: float = 1.0
    timestamp: float = field(default_factory=time)
    embedding: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.embedding is None:
            self.embedding = _embed(self.content)
        self.tags = set(self.tags)


@dataclass
class MemoryStore:
    """Simple vector/tag memory with strength decay."""

    capacity: int = 512
    decay: float = 0.995
    items: List[MemoryItem] = field(default_factory=list)

    def add(self, content: Any, tags: Iterable[str] = (), strength: float = 1.0) -> MemoryItem:
        item = MemoryItem(content=content, tags=set(tags), strength=float(strength))
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda i: (i.strength, i.timestamp), reverse=True)
            del self.items[self.capacity :]
        return item

    remember = add

    def recall(self, query: Any, k: int = 5, tag: str | None = None) -> list[MemoryItem]:
        q = _embed(query)
        candidates = [item for item in self.items if tag is None or tag in item.tags]
        scored = [(_cosine(q, item.embedding) * item.strength, item) for item in candidates]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:k]]

    search = recall

    def consolidate(self) -> None:
        for item in self.items:
            item.strength *= self.decay
        self.items = [item for item in self.items if item.strength > 1e-4]

    def clear(self) -> None:
        self.items.clear()
