"""Memory systems — working memory and episodic (autobiographical) memory.

Working memory: small, fast, decaying — what the brain is thinking about NOW.
Episodic memory: a timeline of everything the brain has experienced,
with importance-based consolidation and forgetting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict


@dataclass
class Episode:
    """One remembered experience."""

    timestamp: float
    kind: str                 # 'sight' | 'sound' | 'thought' | 'fact' | ...
    summary: str
    importance: float = 0.5
    data: dict = field(default_factory=dict)
    recalls: int = 0


class WorkingMemory:
    """Miller's magic number: a handful of active items that decay.

    Items have activation; rehearsing (re-adding) boosts them, time decays
    them, and the weakest item is displaced when capacity is exceeded.
    """

    def __init__(self, capacity: int = 7, decay: float = 0.95):
        self.capacity = capacity
        self.decay = decay
        self._items: dict[str, float] = {}

    def hold(self, item: str, activation: float = 1.0) -> None:
        self.tick()
        current = self._items.get(item, 0.0)
        self._items[item] = min(2.0, current + activation)
        while len(self._items) > self.capacity:
            weakest = min(self._items, key=self._items.get)
            del self._items[weakest]

    def tick(self) -> None:
        """Time passes; everything fades a little."""
        for k in list(self._items):
            self._items[k] *= self.decay
            if self._items[k] < 0.05:
                del self._items[k]

    def contents(self) -> list[tuple[str, float]]:
        return sorted(self._items.items(), key=lambda kv: -kv[1])

    def focus(self) -> str | None:
        """The single most active item — the current focus of thought."""
        if not self._items:
            return None
        return max(self._items, key=self._items.get)

    def __contains__(self, item: str) -> bool:
        return item in self._items


class EpisodicMemory:
    """The brain's autobiographical timeline with consolidation.

    Memories that are important or frequently recalled survive;
    trivial, never-recalled memories are pruned during consolidation
    (the computational analogue of sleep).
    """

    def __init__(self, max_episodes: int = 10000):
        self.max_episodes = max_episodes
        self.episodes: list[Episode] = []

    def record(self, kind: str, summary: str, importance: float = 0.5,
               data: dict | None = None) -> Episode:
        ep = Episode(time.time(), kind, summary, importance, data or {})
        self.episodes.append(ep)
        if len(self.episodes) > self.max_episodes:
            self.consolidate()
        return ep

    def recall(self, query: str, top: int = 5) -> list[Episode]:
        """Cue-based recall: keyword overlap weighted by importance
        and recency. Recalling a memory strengthens it."""
        query_words = set(query.lower().split())
        now = time.time()
        scored = []
        for ep in self.episodes:
            words = set(ep.summary.lower().split())
            words.add(ep.kind.lower())
            overlap = len(query_words & words)
            if overlap == 0:
                continue
            recency = 1.0 / (1.0 + (now - ep.timestamp) / 3600.0)
            score = overlap * (0.5 + ep.importance) * (0.5 + 0.5 * recency)
            scored.append((score, ep))
        scored.sort(key=lambda se: -se[0])
        results = [ep for _, ep in scored[:top]]
        for ep in results:
            ep.recalls += 1
            ep.importance = min(1.0, ep.importance + 0.02)
        return results

    def recent(self, n: int = 10) -> list[Episode]:
        return self.episodes[-n:]

    def consolidate(self, keep_fraction: float = 0.7) -> int:
        """Prune the least valuable memories. Returns how many were forgotten."""
        if not self.episodes:
            return 0
        now = time.time()

        def value(ep: Episode) -> float:
            age_hours = (now - ep.timestamp) / 3600.0
            return ep.importance + 0.1 * ep.recalls - 0.01 * age_hours

        keep = max(1, int(len(self.episodes) * keep_fraction))
        survivors = sorted(self.episodes, key=value, reverse=True)[:keep]
        forgotten = len(self.episodes) - len(survivors)
        survivors.sort(key=lambda e: e.timestamp)
        self.episodes = survivors
        return forgotten

    def __len__(self) -> int:
        return len(self.episodes)

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self.episodes], f)

    @classmethod
    def load(cls, path: str) -> "EpisodicMemory":
        mem = cls()
        with open(path) as f:
            for item in json.load(f):
                mem.episodes.append(Episode(**item))
        return mem
