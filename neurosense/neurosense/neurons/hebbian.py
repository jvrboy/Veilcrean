"""Hebbian learning — 'neurons that fire together, wire together'.

Biologically inspired unsupervised weight adaptation (Oja's rule),
useful for learning correlations and principal components online.
"""

from __future__ import annotations

import numpy as np


class HebbianLayer:
    """A layer that self-organizes with Oja's normalized Hebbian rule.

    After exposure to many input patterns, each output neuron converges
    toward a principal direction of the input distribution.

    >>> layer = HebbianLayer(n_in=10, n_out=3)
    >>> for pattern in data:
    ...     layer.observe(pattern)
    >>> response = layer.respond(new_pattern)
    """

    def __init__(self, n_in: int, n_out: int, lr: float = 0.01,
                 seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(n_out, n_in))
        self.lr = lr
        self.exposures = 0

    def respond(self, x: np.ndarray) -> np.ndarray:
        """Neuron activations for an input pattern."""
        return self.W @ np.asarray(x, dtype=np.float64)

    def observe(self, x: np.ndarray) -> np.ndarray:
        """See a pattern and adapt weights (Oja's rule). Returns response."""
        x = np.asarray(x, dtype=np.float64)
        y = self.W @ x
        # Oja's rule: dW = lr * y * (x - y * W), keeps weights bounded
        self.W += self.lr * (np.outer(y, x) - (y**2)[:, None] * self.W)
        self.exposures += 1
        return y

    def strongest_association(self, x: np.ndarray) -> int:
        """Index of the neuron that responds most strongly to x."""
        return int(np.argmax(np.abs(self.respond(x))))


class AssociativeMemory:
    """Hopfield-style associative memory: store patterns, recall from noise.

    Patterns are bipolar vectors (+1 / -1). Recall converges to the
    stored pattern nearest the noisy cue — content-addressable memory,
    like remembering a whole face from half of it.
    """

    def __init__(self, size: int):
        self.size = size
        self.W = np.zeros((size, size))
        self.stored = 0

    def store(self, pattern: np.ndarray) -> None:
        p = np.sign(np.asarray(pattern, dtype=np.float64))
        p[p == 0] = 1
        self.W += np.outer(p, p) / self.size
        np.fill_diagonal(self.W, 0)
        self.stored += 1

    def recall(self, cue: np.ndarray, max_steps: int = 50) -> np.ndarray:
        s = np.sign(np.asarray(cue, dtype=np.float64))
        s[s == 0] = 1
        for _ in range(max_steps):
            new = np.sign(self.W @ s)
            new[new == 0] = 1
            if np.array_equal(new, s):
                break
            s = new
        return s
