"""Basic neural-network layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .activations import Activation, get_activation


@dataclass
class DenseLayer:
    """Fully connected layer with optional activation."""

    input_size: int
    output_size: int
    activation: str | Activation | None = "linear"
    seed: int | None = None
    weights: np.ndarray = field(init=False)
    bias: np.ndarray = field(init=False)
    _last_input: np.ndarray = field(default=None, init=False, repr=False)
    _last_z: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        rng = np.random.default_rng(self.seed)
        limit = np.sqrt(6.0 / (self.input_size + self.output_size))
        self.weights = rng.uniform(-limit, limit, size=(self.input_size, self.output_size))
        self.bias = np.zeros(self.output_size, dtype=float)
        self.activation = get_activation(self.activation)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"expected {self.input_size} features, got {x.shape[-1]}")
        self._last_input = x
        self._last_z = x @ self.weights + self.bias
        return self.activation(self._last_z)

    __call__ = forward

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if self._last_input is None or self._last_z is None:
            raise RuntimeError("forward must be called before backward")
        grad_output = np.asarray(grad_output, dtype=float)
        grad_z = grad_output * self.activation.derivative(self._last_z)
        batch = max(1, self._last_input.shape[0])
        grad_w = self._last_input.T @ grad_z / batch
        grad_b = grad_z.mean(axis=0)
        grad_input = grad_z @ self.weights.T
        return grad_input, {"weights": grad_w, "bias": grad_b}

    def apply_gradients(self, grads: Dict[str, np.ndarray], lr: float) -> None:
        self.weights -= lr * grads.get("weights", 0.0)
        self.bias -= lr * grads.get("bias", 0.0)

    def parameters(self) -> Dict[str, np.ndarray]:
        return {"weights": self.weights, "bias": self.bias}


@dataclass
class DropoutLayer:
    """Inverted dropout for dense inputs."""

    rate: float = 0.5
    seed: int | None = None
    training: bool = True
    _mask: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate < 1.0:
            raise ValueError("rate must be in [0, 1)")
        self._rng = np.random.default_rng(self.seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not self.training or self.rate == 0.0:
            self._mask = np.ones_like(x)
            return x
        keep = 1.0 - self.rate
        self._mask = (self._rng.random(x.shape) < keep).astype(float) / keep
        return x * self._mask

    __call__ = forward

    def backward(self, grad_output: np.ndarray):
        return np.asarray(grad_output) * (1.0 if self._mask is None else self._mask), {}

    def apply_gradients(self, grads, lr: float) -> None:
        return None


@dataclass
class LayerNorm:
    """Feature-wise layer normalization."""

    eps: float = 1e-5

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + self.eps)

    __call__ = forward

    def backward(self, grad_output: np.ndarray):
        # For lightweight use we pass gradients through; the layer is typically
        # used for inference in this package.
        return np.asarray(grad_output, dtype=float), {}

    def apply_gradients(self, grads, lr: float) -> None:
        return None
