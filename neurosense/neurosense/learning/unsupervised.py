"""Unsupervised learning — discovering structure without a teacher.

KMeans clustering and a Self-Organizing Map (Kohonen network),
both from scratch with numpy.
"""

from __future__ import annotations

import numpy as np


class KMeans:
    """K-means clustering with k-means++ initialization.

    >>> km = KMeans(k=3).fit(X)
    >>> km.predict(x)
    """

    def __init__(self, k: int, max_iters: int = 200, tol: float = 1e-6,
                 seed: int | None = None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.centroids: np.ndarray | None = None
        self.inertia_: float = float("inf")

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self.centroids = self._init_pp(X)
        for _ in range(self.max_iters):
            labels = self._assign(X)
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if (labels == i).any()
                else X[self.rng.integers(len(X))]
                for i in range(self.k)
            ])
            shift = float(np.linalg.norm(new_centroids - self.centroids))
            self.centroids = new_centroids
            if shift < self.tol:
                break
        labels = self._assign(X)
        self.inertia_ = float(sum(
            np.linalg.norm(X[labels == i] - self.centroids[i], axis=1).sum()
            for i in range(self.k)))
        return self

    def _init_pp(self, X: np.ndarray) -> np.ndarray:
        """k-means++ seeding: spread initial centroids far apart."""
        centroids = [X[self.rng.integers(len(X))]]
        for _ in range(1, self.k):
            d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids],
                        axis=0)
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(len(X), 1 / len(X))
            centroids.append(X[self.rng.choice(len(X), p=probs)])
        return np.array(centroids)

    def _assign(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :],
                               axis=2)
        return np.argmin(dists, axis=1)

    def predict(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=np.float64)
        return int(np.argmin(np.linalg.norm(self.centroids - x, axis=1)))


class SelfOrganizingMap:
    """Kohonen SOM: a 2-D sheet of neurons that topologically organizes
    high-dimensional input — a computational model of cortical maps.

    >>> som = SelfOrganizingMap(width=8, height=8, dim=16)
    >>> som.fit(X, epochs=20)
    >>> som.locate(x)   # -> (row, col) of the best-matching neuron
    """

    def __init__(self, width: int, height: int, dim: int,
                 seed: int | None = None):
        self.width, self.height, self.dim = width, height, dim
        rng = np.random.default_rng(seed)
        self.weights = rng.random((height, width, dim))
        rows, cols = np.mgrid[0:height, 0:width]
        self._grid = np.stack([rows, cols], axis=-1).astype(np.float64)

    def locate(self, x: np.ndarray) -> tuple[int, int]:
        """Best-matching unit for an input vector."""
        d = np.linalg.norm(self.weights - np.asarray(x, dtype=np.float64),
                           axis=2)
        idx = np.unravel_index(np.argmin(d), d.shape)
        return int(idx[0]), int(idx[1])

    def fit(self, X: np.ndarray, epochs: int = 20, lr0: float = 0.5,
            radius0: float | None = None) -> "SelfOrganizingMap":
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        radius0 = radius0 or max(self.width, self.height) / 2
        total_steps = epochs * len(X)
        step = 0
        rng = np.random.default_rng()
        for _ in range(epochs):
            for i in rng.permutation(len(X)):
                x = X[i]
                progress = step / max(total_steps, 1)
                lr = lr0 * np.exp(-3 * progress)
                radius = max(radius0 * np.exp(-3 * progress), 0.5)
                bmu = np.array(self.locate(x), dtype=np.float64)
                dist2 = np.sum((self._grid - bmu) ** 2, axis=-1)
                influence = np.exp(-dist2 / (2 * radius**2))
                self.weights += (lr * influence[..., None]
                                 * (x - self.weights))
                step += 1
        return self

    def umatrix(self) -> np.ndarray:
        """Average distance of each neuron to its neighbors — reveals
        cluster boundaries as ridges."""
        u = np.zeros((self.height, self.width))
        for r in range(self.height):
            for c in range(self.width):
                dists = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        dists.append(np.linalg.norm(
                            self.weights[r, c] - self.weights[nr, nc]))
                u[r, c] = np.mean(dists) if dists else 0.0
        return u
