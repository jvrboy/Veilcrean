"""Activation functions and their derivatives, implemented from scratch."""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_prime(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def linear(x: np.ndarray) -> np.ndarray:
    return x


def linear_prime(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def leaky_relu_prime(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


ACTIVATIONS = {
    "relu": (relu, relu_prime),
    "sigmoid": (sigmoid, sigmoid_prime),
    "tanh": (tanh, tanh_prime),
    "linear": (linear, linear_prime),
    "leaky_relu": (leaky_relu, leaky_relu_prime),
}
