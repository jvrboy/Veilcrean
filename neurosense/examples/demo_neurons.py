"""Train a tiny network on XOR-like data."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neurosense.neurons import DenseLayer, NeuralNetwork


def main() -> None:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    net = NeuralNetwork.from_sizes([2, 4, 1], activation="tanh", output_activation="sigmoid", seed=7)
    history = net.fit(x, y, epochs=10, lr=0.1)
    print("loss", round(history[-1], 4))
    print(net.predict(x).round(3))


if __name__ == "__main__":
    main()
