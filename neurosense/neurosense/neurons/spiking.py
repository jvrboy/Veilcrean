"""Spiking neurons — leaky integrate-and-fire model.

The most biologically realistic neuron model in the library: membrane
potential integrates input current, leaks over time, and fires a spike
when it crosses threshold, followed by a refractory period.
"""

from __future__ import annotations

import numpy as np


class SpikingNeuron:
    """A single leaky integrate-and-fire neuron.

    >>> n = SpikingNeuron()
    >>> spikes = [n.step(current=1.6) for _ in range(100)]
    """

    def __init__(self, threshold: float = 1.0, leak: float = 0.9,
                 rest: float = 0.0, refractory_steps: int = 3):
        self.threshold = threshold
        self.leak = leak
        self.rest = rest
        self.refractory_steps = refractory_steps
        self.potential = rest
        self._refractory = 0
        self.spike_count = 0

    def step(self, current: float) -> bool:
        """Advance one timestep with input current. Returns True on spike."""
        if self._refractory > 0:
            self._refractory -= 1
            self.potential = self.rest
            return False
        self.potential = self.leak * self.potential + current
        if self.potential >= self.threshold:
            self.potential = self.rest
            self._refractory = self.refractory_steps
            self.spike_count += 1
            return True
        return False

    def reset(self) -> None:
        self.potential = self.rest
        self._refractory = 0
        self.spike_count = 0


class SpikingNetwork:
    """A recurrently connected population of spiking neurons with STDP-like
    plasticity: synapses that helped cause a spike are strengthened.

    >>> net = SpikingNetwork(n_neurons=50, seed=0)
    >>> for t in range(200):
    ...     spikes = net.step(external=stimulus_vector)
    """

    def __init__(self, n_neurons: int, connectivity: float = 0.2,
                 plasticity: float = 0.005, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.n = n_neurons
        mask = rng.random((n_neurons, n_neurons)) < connectivity
        np.fill_diagonal(mask, False)
        self.W = np.where(mask, rng.normal(0.3, 0.1, (n_neurons, n_neurons)), 0.0)
        self.neurons = [SpikingNeuron() for _ in range(n_neurons)]
        self.plasticity = plasticity
        self._last_spikes = np.zeros(n_neurons, dtype=bool)

    def step(self, external: np.ndarray | None = None) -> np.ndarray:
        """One timestep: propagate spikes, integrate, fire, adapt synapses."""
        recurrent = self.W @ self._last_spikes.astype(np.float64)
        drive = recurrent + (np.asarray(external, dtype=np.float64)
                             if external is not None else 0.0)
        spikes = np.array([n.step(float(c))
                           for n, c in zip(self.neurons, drive)])
        # STDP-like: strengthen synapse pre->post when pre fired last step
        # and post fired now; mild decay everywhere for stability.
        if self._last_spikes.any() and spikes.any():
            self.W[np.ix_(spikes, self._last_spikes)] += self.plasticity
        self.W *= (1.0 - self.plasticity * 0.01)
        np.clip(self.W, -2.0, 2.0, out=self.W)
        self._last_spikes = spikes
        return spikes

    def firing_rates(self, steps_run: int) -> np.ndarray:
        """Average firing rate of each neuron over the run so far."""
        return np.array([n.spike_count / max(steps_run, 1)
                         for n in self.neurons])
