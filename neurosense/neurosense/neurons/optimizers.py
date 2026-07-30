"""Gradient-descent optimizers implemented from scratch."""

from __future__ import annotations

import numpy as np


class Optimizer:
    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Plain stochastic gradient descent."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum(Optimizer):
    """SGD with classical momentum."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self._v: dict[int, np.ndarray] = {}

    def step(self, params, grads):
        for p, g in zip(params, grads):
            key = id(p)
            v = self._v.setdefault(key, np.zeros_like(p))
            v *= self.beta
            v -= self.lr * g
            p += v


class Adam(Optimizer):
    """Adaptive moment estimation (Adam)."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self._t = 0

    def step(self, params, grads):
        self._t += 1
        for p, g in zip(params, grads):
            key = id(p)
            m = self._m.setdefault(key, np.zeros_like(p))
            v = self._v.setdefault(key, np.zeros_like(p))
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g**2
            m_hat = m / (1 - self.beta1**self._t)
            v_hat = v / (1 - self.beta2**self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
