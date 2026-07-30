"""The Eye — high-level visual perception organ.

The Eye converts raw pixel arrays into structured VisualPercepts and can
learn to recognize what it has seen before (nearest-signature recall).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import (
    to_grayscale,
    sobel_edges,
    detect_corners,
    find_blobs,
    image_signature,
)


@dataclass
class VisualPercept:
    """A structured description of what the eye saw."""

    signature: np.ndarray
    brightness: float
    contrast: float
    edge_density: float
    dominant_orientation: float
    corners: list = field(default_factory=list)
    blobs: list = field(default_factory=list)
    label: str | None = None
    confidence: float = 0.0

    def describe(self) -> str:
        parts = []
        parts.append("bright scene" if self.brightness > 0.6
                     else "dark scene" if self.brightness < 0.3
                     else "moderately lit scene")
        parts.append(f"{len(self.blobs)} distinct object region(s)")
        parts.append(f"{len(self.corners)} corner feature(s)")
        if self.edge_density > 0.15:
            parts.append("high visual complexity")
        elif self.edge_density < 0.03:
            parts.append("smooth / uniform texture")
        if self.label:
            parts.append(f"recognized as '{self.label}' "
                         f"({self.confidence:.0%} confidence)")
        return "I see a " + ", ".join(parts) + "."


class Eye:
    """Visual perception with recognition memory.

    >>> eye = Eye()
    >>> eye.memorize(circle_image, "circle")
    >>> percept = eye.perceive(new_image)
    >>> percept.label   # 'circle' if it looks similar
    """

    def __init__(self, recognition_threshold: float = 0.85):
        self.recognition_threshold = recognition_threshold
        self._memory: list[tuple[np.ndarray, str]] = []

    # ------------------------------------------------------------------ #
    def perceive(self, image: np.ndarray) -> VisualPercept:
        """Look at an image and produce a full structured percept."""
        gray = to_grayscale(image)
        mag, ori = sobel_edges(gray)
        edge_mask = mag > 0.2
        signature = image_signature(image)

        if edge_mask.any():
            weights = mag[edge_mask]
            angles = ori[edge_mask]
            dominant = float(np.average(angles, weights=weights))
        else:
            dominant = 0.0

        percept = VisualPercept(
            signature=signature,
            brightness=float(gray.mean()),
            contrast=float(gray.std()),
            edge_density=float(edge_mask.mean()),
            dominant_orientation=dominant,
            corners=detect_corners(gray),
            blobs=find_blobs(gray, threshold=float(gray.mean())),
        )
        label, conf = self._recognize(signature)
        if label is not None:
            percept.label = label
            percept.confidence = conf
        return percept

    # ------------------------------------------------------------------ #
    def memorize(self, image: np.ndarray, label: str) -> None:
        """Associate an image with a label — one-shot visual learning."""
        self._memory.append((image_signature(image), label))

    def _recognize(self, signature: np.ndarray) -> tuple[str | None, float]:
        best_label, best_sim = None, 0.0
        for stored, label in self._memory:
            sim = _cosine(signature, stored)
            if sim > best_sim:
                best_label, best_sim = label, sim
        if best_sim >= self.recognition_threshold:
            return best_label, best_sim
        return None, best_sim

    def known_labels(self) -> list[str]:
        return sorted({label for _, label in self._memory})


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))
