"""Tiny reinforcement-learning utilities."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Hashable, Iterable

import numpy as np


def epsilon_greedy(values: Iterable[float], epsilon: float = 0.1, seed: int | None = None) -> int:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        raise ValueError("values must be non-empty")
    rng = np.random.default_rng(seed)
    if rng.random() < epsilon:
        return int(rng.integers(0, values.size))
    return int(np.argmax(values))


@dataclass
class ReplayBuffer:
    capacity: int = 1000
    data: Deque[Any] = field(default_factory=deque, init=False)

    def push(self, transition: Any) -> None:
        if len(self.data) >= self.capacity:
            self.data.popleft()
        self.data.append(transition)

    append = push

    def sample(self, batch_size: int, seed: int | None = None) -> list[Any]:
        rng = np.random.default_rng(seed)
        n = min(batch_size, len(self.data))
        if n <= 0:
            return []
        idx = rng.choice(len(self.data), n, replace=False)
        items = list(self.data)
        return [items[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class QLearningAgent:
    actions: list[Hashable]
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.1
    q: dict[tuple[Hashable, Hashable], float] = field(default_factory=dict)

    def values(self, state: Hashable) -> np.ndarray:
        return np.array([self.q.get((state, action), 0.0) for action in self.actions], dtype=float)

    def act(self, state: Hashable, seed: int | None = None) -> Hashable:
        idx = epsilon_greedy(self.values(state), self.epsilon, seed=seed)
        return self.actions[idx]

    def update(self, state: Hashable, action: Hashable, reward: float, next_state: Hashable, done: bool = False) -> float:
        old = self.q.get((state, action), 0.0)
        target = float(reward) if done else float(reward) + self.gamma * float(np.max(self.values(next_state)))
        new = old + self.alpha * (target - old)
        self.q[(state, action)] = new
        return new

    learn = update
