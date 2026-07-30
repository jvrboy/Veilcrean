"""Leaky integrate-and-fire spiking neurons."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np


def poisson_encode(values: Iterable[float], steps: int = 10, max_rate: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Encode values in ``[0, 1]`` as Poisson spike trains."""
    vals = np.clip(np.asarray(list(values), dtype=float), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    return (rng.random((steps, vals.size)) < vals * max_rate).astype(float)


@dataclass
class LIFNeuron:
    threshold: float = 1.0
    decay: float = 0.9
    reset: float = 0.0
    potential: float = 0.0

    def step(self, current: float) -> int:
        self.potential = self.decay * self.potential + float(current)
        if self.potential >= self.threshold:
            self.potential = self.reset
            return 1
        return 0

    def reset_state(self) -> None:
        self.potential = self.reset


@dataclass
class SpikingNetwork:
    neurons: List[LIFNeuron] = field(default_factory=list)

    def step(self, currents: Iterable[float]) -> np.ndarray:
        currents = list(currents)
        if not self.neurons:
            self.neurons = [LIFNeuron() for _ in currents]
        if len(currents) != len(self.neurons):
            raise ValueError("current count must match neuron count")
        return np.array([neuron.step(current) for neuron, current in zip(self.neurons, currents)], dtype=int)

    def run(self, current_sequence: Iterable[Iterable[float]]) -> np.ndarray:
        return np.stack([self.step(currents) for currents in current_sequence], axis=0)
