"""Neurons demo: backprop network, Hebbian learning, associative memory,
spiking neurons.

Run:  python examples/demo_neurons.py
"""

import numpy as np

from neurosense.neurons import (
    NeuralNetwork, HebbianLayer, SpikingNetwork,
)
from neurosense.neurons.hebbian import AssociativeMemory


def main():
    rng = np.random.default_rng(42)

    # ---- 1. Backprop network learns XOR ----
    print("--- BACKPROP: XOR ---")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)  # one-hot
    net = NeuralNetwork([2, 8, 2], activation="tanh", output="softmax", seed=1)
    net.train(X, Y, epochs=800, batch_size=4)
    for x in X:
        print(f"  {x} -> class {net.predict_class(x)}")

    # ---- 2. Hebbian layer discovers input correlation ----
    print("\n--- HEBBIAN: unsupervised correlation ---")
    hebb = HebbianLayer(n_in=5, n_out=2, seed=0)
    direction = np.array([1.0, 1.0, 0.0, -1.0, -1.0])
    for _ in range(500):
        hebb.observe(direction * rng.normal(1, 0.2) + rng.normal(0, 0.05, 5))
    print("  learned weight vector ~ input direction:",
          np.round(hebb.W[0] / np.abs(hebb.W[0]).max(), 2))

    # ---- 3. Associative (Hopfield) memory recalls from noise ----
    print("\n--- ASSOCIATIVE MEMORY: recall from noisy cue ---")
    mem = AssociativeMemory(size=16)
    pattern = np.array([1, 1, 1, 1, -1, -1, -1, -1] * 2, dtype=float)
    mem.store(pattern)
    noisy = pattern.copy()
    noisy[[0, 5, 9]] *= -1  # corrupt 3 bits
    recalled = mem.recall(noisy)
    print("  recovered original:", bool(np.array_equal(recalled, pattern)))

    # ---- 4. Spiking network dynamics ----
    print("\n--- SPIKING NETWORK: leaky integrate-and-fire ---")
    snn = SpikingNetwork(n_neurons=30, seed=3)
    stimulus = np.zeros(30)
    stimulus[:5] = 1.2  # drive the first 5 neurons
    steps = 100
    for _ in range(steps):
        snn.step(external=stimulus)
    rates = snn.firing_rates(steps)
    print(f"  driven neurons fire at {rates[:5].mean():.2f}, "
          f"others at {rates[5:].mean():.2f} (activity spread via synapses)")


if __name__ == "__main__":
    main()
