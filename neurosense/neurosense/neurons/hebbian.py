"""Hebbian and Oja learning utilities."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def hebbian_update(weights: np.ndarray, pre: np.ndarray, post: np.ndarray, lr: float = 0.01) -> np.ndarray:
    """Classic Hebbian outer-product update."""
    weights = np.asarray(weights, dtype=float)
    return weights + lr * np.outer(np.asarray(pre, dtype=float), np.asarray(post, dtype=float))


def oja_update(weights: np.ndarray, pre: np.ndarray, post: np.ndarray, lr: float = 0.01) -> np.ndarray:
    """Oja's normalized Hebbian update."""
    weights = np.asarray(weights, dtype=float)
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    return weights + lr * (np.outer(pre, post) - weights * np.square(post))


@dataclass
class HebbianSynapse:
    weight: float = 0.0
    lr: float = 0.01

    def learn(self, pre: float, post: float) -> float:
        self.weight += self.lr * float(pre) * float(post)
        return self.weight


@dataclass
class HebbianNetwork:
    input_size: int
    output_size: int
    lr: float = 0.01
    weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.weights = np.zeros((self.input_size, self.output_size), dtype=float)

    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.weights

    def learn(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.weights = hebbian_update(self.weights, x, y, self.lr)
        return self.weights
