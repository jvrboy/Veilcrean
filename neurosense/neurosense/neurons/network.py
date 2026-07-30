"""Sequential neural network implemented with NumPy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np

from .activations import softmax
from .layers import DenseLayer


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, np.ndarray]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    diff = y_pred - y_true
    return float(np.mean(diff * diff)), 2.0 * diff / max(1, diff.size)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    logits = np.asarray(logits, dtype=float)
    probs = softmax(logits, axis=-1)
    labels = np.asarray(labels)
    if labels.ndim == 1:
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(labels)), labels.astype(int)] = 1.0
    else:
        onehot = labels.astype(float)
    loss = -float(np.mean(np.sum(onehot * np.log(probs + 1e-12), axis=-1)))
    grad = (probs - onehot) / max(1, logits.shape[0])
    return loss, grad


@dataclass
class NeuralNetwork:
    """A tiny sequential neural network."""

    layers: List[object] = field(default_factory=list)

    def add(self, layer: object) -> "NeuralNetwork":
        self.layers.append(layer)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=float)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    predict = forward
    __call__ = forward

    def backward(self, grad: np.ndarray, lr: float = 0.01) -> np.ndarray:
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                grad, grads = layer.backward(grad)
                if hasattr(layer, "apply_gradients"):
                    layer.apply_gradients(grads, lr)
        return grad

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01, loss: str = "mse") -> float:
        pred = self.forward(x)
        if loss == "mse":
            value, grad = mse_loss(pred, y)
        elif loss in {"ce", "cross_entropy"}:
            value, grad = cross_entropy_loss(pred, y)
        else:
            raise ValueError("loss must be 'mse' or 'cross_entropy'")
        self.backward(grad, lr=lr)
        return value

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01, loss: str = "mse") -> list[float]:
        history: list[float] = []
        for _ in range(epochs):
            history.append(self.train_step(x, y, lr=lr, loss=loss))
        return history

    @classmethod
    def from_sizes(cls, sizes: Iterable[int], activation: str = "relu", output_activation: str = "linear", seed: int | None = None) -> "NeuralNetwork":
        sizes = list(sizes)
        if len(sizes) < 2:
            raise ValueError("at least input and output sizes are required")
        network = cls()
        rng = np.random.default_rng(seed)
        for idx, (a, b) in enumerate(zip(sizes, sizes[1:])):
            act = output_activation if idx == len(sizes) - 2 else activation
            network.add(DenseLayer(a, b, activation=act, seed=int(rng.integers(0, 2**31 - 1))))
        return network
