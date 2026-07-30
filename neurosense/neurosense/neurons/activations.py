"""Activation functions and derivatives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

ArrayFn = Callable[[np.ndarray], np.ndarray]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out = np.empty_like(x, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    y = np.tanh(x)
    return 1.0 - y * y


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) > 0.0).astype(float)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(np.asarray(x) > 0.0, x, alpha * np.asarray(x))


def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(np.asarray(x) > 0.0, 1.0, alpha)


def linear(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(x, dtype=float))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass(frozen=True)
class Activation:
    name: str
    function: ArrayFn
    derivative: ArrayFn

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.function(x)


_ACTIVATIONS: Dict[str, Activation] = {
    "sigmoid": Activation("sigmoid", sigmoid, sigmoid_derivative),
    "tanh": Activation("tanh", tanh, tanh_derivative),
    "relu": Activation("relu", relu, relu_derivative),
    "leaky_relu": Activation("leaky_relu", leaky_relu, leaky_relu_derivative),
    "linear": Activation("linear", linear, linear_derivative),
}


def get_activation(name: str | Activation | None) -> Activation:
    """Return an activation by name; defaults to linear."""
    if isinstance(name, Activation):
        return name
    key = "linear" if name is None else str(name).lower()
    if key not in _ACTIVATIONS:
        raise KeyError(f"unknown activation {name!r}")
    return _ACTIVATIONS[key]
