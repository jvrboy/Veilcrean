"""Unsupervised learning algorithms implemented with NumPy."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def normalize(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 1e-12 else float(np.dot(a, b) / denom)


@dataclass
class PCA:
    n_components: int = 2
    mean_: np.ndarray | None = field(default=None, init=False)
    components_: np.ndarray | None = field(default=None, init=False)
    explained_variance_: np.ndarray | None = field(default=None, init=False)

    def fit(self, x: np.ndarray) -> "PCA":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("PCA expects a 2-D array")
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.n_components]
        denom = max(1, x.shape[0] - 1)
        self.explained_variance_ = (s[: self.n_components] ** 2) / denom
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("fit must be called before transform")
        return (np.asarray(x, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


@dataclass
class KMeans:
    n_clusters: int = 2
    max_iter: int = 100
    seed: int | None = None
    centers_: np.ndarray | None = field(default=None, init=False)
    labels_: np.ndarray | None = field(default=None, init=False)

    def fit(self, x: np.ndarray) -> "KMeans":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("KMeans expects a 2-D array")
        if not 1 <= self.n_clusters <= len(x):
            raise ValueError("n_clusters must be between 1 and number of samples")
        rng = np.random.default_rng(self.seed)
        centers = x[rng.choice(len(x), self.n_clusters, replace=False)].copy()
        labels = np.zeros(len(x), dtype=int)
        for _ in range(self.max_iter):
            dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1)
            new_labels = np.argmin(dist, axis=1)
            if np.array_equal(new_labels, labels) and self.centers_ is not None:
                break
            labels = new_labels
            for k in range(self.n_clusters):
                if np.any(labels == k):
                    centers[k] = x[labels == k].mean(axis=0)
        self.centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("fit must be called before predict")
        x = np.asarray(x, dtype=float)
        dist = np.linalg.norm(x[:, None, :] - self.centers_[None, :, :], axis=-1)
        return np.argmin(dist, axis=1)

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).labels_


def kmeans(x: np.ndarray, n_clusters: int = 2, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, seed=seed).fit(x)
    return model.centers_, model.labels_
