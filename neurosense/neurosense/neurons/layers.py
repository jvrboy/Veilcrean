"""Neural network layers with manual forward/backward passes (backprop)."""

from __future__ import annotations

import numpy as np

from .activations import ACTIVATIONS


class Layer:
    """Base layer interface."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def params(self) -> list[np.ndarray]:
        return []

    @property
    def grads(self) -> list[np.ndarray]:
        return []


class Dense(Layer):
    """Fully connected layer: y = xW + b, with He/Xavier initialization."""

    def __init__(self, n_in: int, n_out: int, init: str = "he",
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        if init == "he":
            scale = np.sqrt(2.0 / n_in)
        else:  # xavier
            scale = np.sqrt(1.0 / n_in)
        self.W = rng.normal(0.0, scale, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.dW[...] = self._x.T @ grad
        self.db[...] = grad.sum(axis=0)
        return grad @ self.W.T

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class Activation(Layer):
    """Element-wise nonlinearity layer. name in ACTIVATIONS."""

    def __init__(self, name: str):
        if name not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{name}'. "
                             f"Choose from {list(ACTIVATIONS)}")
        self.name = name
        self.fn, self.fn_prime = ACTIVATIONS[name]
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return self.fn(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.fn_prime(self._x)


class Dropout(Layer):
    """Inverted dropout for regularization. Call .train()/.eval() to switch."""

    def __init__(self, rate: float = 0.5, rng: np.random.Generator | None = None):
        self.rate = rate
        self.rng = rng or np.random.default_rng()
        self.training = True
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0:
            return x
        self._mask = (self.rng.random(x.shape) > self.rate) / (1 - self.rate)
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0:
            return grad
        return grad * self._mask
