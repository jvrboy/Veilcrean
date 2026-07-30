"""Feature extraction for simple visual inputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class VisualFeatures:
    """Compact summary of a grayscale image."""

    brightness: float
    contrast: float
    edge_density: float
    symmetry: float
    centroid_x: float
    centroid_y: float
    width: int
    height: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _as_array(image: Any) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.size == 0:
        raise ValueError("image must contain at least one pixel")
    if arr.ndim not in (2, 3):
        raise ValueError("image must be a 2-D grayscale or 3-D color array")
    return arr


def to_grayscale(image: Any) -> np.ndarray:
    """Convert an image-like object to a 2-D grayscale float array."""
    arr = _as_array(image)
    if arr.ndim == 2:
        return arr.astype(float, copy=False)
    channels = arr.shape[-1]
    if channels == 1:
        return arr[..., 0]
    # Rec. 709 luma coefficients, truncated/extended for unusual channel counts.
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=float)
    if channels < 3:
        weights = np.ones(channels, dtype=float) / channels
    else:
        weights = np.pad(weights, (0, channels - 3), constant_values=0.0)
        weights = weights / weights.sum()
    return np.tensordot(arr, weights[:channels], axes=([-1], [0]))


def normalize_image(image: Any) -> np.ndarray:
    """Normalize pixels to the ``[0, 1]`` range."""
    gray = to_grayscale(image)
    mn = float(np.nanmin(gray))
    mx = float(np.nanmax(gray))
    if not np.isfinite(mn) or not np.isfinite(mx):
        raise ValueError("image contains no finite values")
    if mx <= mn:
        return np.zeros_like(gray, dtype=float)
    return (gray - mn) / (mx - mn)


def edge_magnitude(image: Any) -> np.ndarray:
    """Return a Sobel-like gradient magnitude image."""
    gray = normalize_image(image)
    gy, gx = np.gradient(gray)
    return np.hypot(gx, gy)


def _centroid(gray: np.ndarray) -> Tuple[float, float]:
    weights = np.clip(gray, 0.0, None)
    total = float(weights.sum())
    h, w = gray.shape
    if total <= 1e-12:
        return 0.5, 0.5
    yy, xx = np.indices(gray.shape)
    cx = float((xx * weights).sum() / total) / max(w - 1, 1)
    cy = float((yy * weights).sum() / total) / max(h - 1, 1)
    return cx, cy


def extract_visual_features(image: Any) -> VisualFeatures:
    """Extract brightness, contrast, edge, symmetry, and centroid features."""
    gray = normalize_image(image)
    edges = edge_magnitude(gray)
    h, w = gray.shape
    left = gray[:, : w // 2]
    right = np.fliplr(gray[:, w - w // 2 :])
    if left.size and right.size:
        n = min(left.shape[1], right.shape[1])
        asym = float(np.mean(np.abs(left[:, :n] - right[:, :n])))
        symmetry = 1.0 - min(1.0, asym)
    else:
        symmetry = 1.0
    cx, cy = _centroid(gray)
    return VisualFeatures(
        brightness=float(np.mean(gray)),
        contrast=float(np.std(gray)),
        edge_density=float(np.mean(edges > (edges.mean() + edges.std()))),
        symmetry=float(symmetry),
        centroid_x=float(cx),
        centroid_y=float(cy),
        width=int(w),
        height=int(h),
    )


class EyeFeatureExtractor:
    """State-less visual feature extractor class."""

    def transform(self, image: Any) -> Dict[str, float]:
        return extract_visual_features(image).as_dict()

    def batch_transform(self, images: Iterable[Any]) -> list[Dict[str, float]]:
        return [self.transform(image) for image in images]
