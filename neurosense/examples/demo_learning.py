"""Demonstrate unsupervised and reinforcement helpers."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neurosense.learning import KMeans, PCA, QLearningAgent


def main() -> None:
    x = np.array([[0, 0], [0.1, 0], [3, 3], [3.2, 2.9]], dtype=float)
    labels = KMeans(n_clusters=2, seed=1).fit_predict(x)
    reduced = PCA(n_components=1).fit_transform(x)
    agent = QLearningAgent(actions=["left", "right"], epsilon=0.0)
    agent.update("start", "right", 1.0, "end", done=True)
    print("clusters", labels.tolist())
    print("pca", reduced.round(3).reshape(-1).tolist())
    print("best action", agent.act("start"))


if __name__ == "__main__":
    main()
