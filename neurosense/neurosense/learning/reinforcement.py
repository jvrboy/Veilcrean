"""Reinforcement learning — learning from consequences.

Tabular Q-learning with epsilon-greedy exploration and decay.
The brain uses this to learn which actions lead to good outcomes.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict


class QLearner:
    """Tabular Q-learning agent for any hashable state / action space.

    >>> agent = QLearner(actions=["left", "right"])
    >>> a = agent.choose(state)
    >>> agent.learn(state, a, reward, next_state)
    """

    def __init__(self, actions: list, lr: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995, seed: int | None = None):
        self.actions = list(actions)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = random.Random(seed)
        self.Q: dict = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.total_updates = 0

    # ------------------------------------------------------------------ #
    def choose(self, state, explore: bool = True):
        """Pick an action: epsilon-greedy exploration vs exploitation."""
        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)
        return self.best_action(state)

    def best_action(self, state):
        values = self.Q[self._key(state)]
        best = max(values.values())
        candidates = [a for a, v in values.items() if v == best]
        return self._rng.choice(candidates)

    # ------------------------------------------------------------------ #
    def learn(self, state, action, reward: float, next_state,
              done: bool = False) -> float:
        """Q-learning update. Returns the new Q(s, a)."""
        key, next_key = self._key(state), self._key(next_state)
        future = 0.0 if done else max(self.Q[next_key].values())
        target = reward + self.gamma * future
        self.Q[key][action] += self.lr * (target - self.Q[key][action])
        self.total_updates += 1
        if done:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)
        return self.Q[key][action]

    def value_of(self, state) -> float:
        return max(self.Q[self._key(state)].values())

    @staticmethod
    def _key(state):
        if isinstance(state, (list,)):
            return tuple(state)
        try:
            hash(state)
            return state
        except TypeError:
            return str(state)

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"actions": self.actions, "epsilon": self.epsilon,
                       "Q": {str(k): v for k, v in self.Q.items()}}, f)

    def load_table(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.epsilon = data["epsilon"]
        for k, v in data["Q"].items():
            self.Q[k] = v
