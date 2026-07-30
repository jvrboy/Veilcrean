"""Small optimizers for NumPy parameter arrays."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, MutableMapping, Sequence

import numpy as np


def _iter_param_grad(params, grads):
    if isinstance(params, MutableMapping):
        for key, param in params.items():
            yield key, param, grads[key]
    else:
        for idx, (param, grad) in enumerate(zip(params, grads)):
            yield idx, param, grad


@dataclass
class SGD:
    lr: float = 0.01
    momentum: float = 0.0
    _velocity: dict = field(default_factory=dict, init=False, repr=False)

    def step(self, params, grads=None) -> None:
        """Update arrays in-place.

        Accepts either ``step({name: array}, {name: grad})`` or
        ``step([(array, grad), ...])``.
        """
        if grads is None:
            pairs = [(idx, p, g) for idx, (p, g) in enumerate(params)]
        else:
            pairs = list(_iter_param_grad(params, grads))
        for key, param, grad in pairs:
            if self.momentum:
                v = self._velocity.get(key, np.zeros_like(param))
                v = self.momentum * v + grad
                self._velocity[key] = v
                grad = v
            param -= self.lr * grad


@dataclass
class Adam:
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    _m: dict = field(default_factory=dict, init=False, repr=False)
    _v: dict = field(default_factory=dict, init=False, repr=False)
    _t: int = field(default=0, init=False, repr=False)

    def step(self, params, grads=None) -> None:
        if grads is None:
            pairs = [(idx, p, g) for idx, (p, g) in enumerate(params)]
        else:
            pairs = list(_iter_param_grad(params, grads))
        self._t += 1
        for key, param, grad in pairs:
            m = self._m.get(key, np.zeros_like(param))
            v = self._v.get(key, np.zeros_like(param))
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
            self._m[key] = m
            self._v[key] = v
            m_hat = m / (1.0 - self.beta1**self._t)
            v_hat = v / (1.0 - self.beta2**self._t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
