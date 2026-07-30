"""High-level visual perception."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .features import extract_visual_features, normalize_image


def detect_motion(previous: Any, current: Any, threshold: float = 0.08) -> Dict[str, float | bool]:
    """Estimate frame-to-frame motion with normalized absolute difference."""
    prev = normalize_image(previous)
    cur = normalize_image(current)
    if prev.shape != cur.shape:
        raise ValueError("previous and current frames must have the same shape")
    diff = np.abs(cur - prev)
    score = float(diff.mean())
    return {
        "motion": bool(score >= threshold),
        "motion_score": score,
        "changed_fraction": float(np.mean(diff >= threshold)),
    }


def summarize_scene(image: Any) -> str:
    """Create a short human-readable scene summary from visual features."""
    f = extract_visual_features(image)
    light = "bright" if f.brightness > 0.66 else "dim" if f.brightness < 0.33 else "balanced"
    texture = "high-detail" if f.edge_density > 0.2 else "smooth"
    focus = "centered" if abs(f.centroid_x - 0.5) < 0.15 and abs(f.centroid_y - 0.5) < 0.15 else "off-center"
    return f"{light}, {texture}, {focus} scene ({f.width}x{f.height})"


@dataclass
class VisionSensor:
    """Perceive visual frames and retain the last frame for motion cues."""

    motion_threshold: float = 0.08
    previous_frame: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def perceive(self, image: Any) -> Dict[str, Any]:
        features = extract_visual_features(image).as_dict()
        result: Dict[str, Any] = {
            "modality": "vision",
            "features": features,
            "summary": summarize_scene(image),
        }
        current = normalize_image(image)
        if self.previous_frame is not None and self.previous_frame.shape == current.shape:
            result.update(detect_motion(self.previous_frame, current, self.motion_threshold))
        else:
            result.update({"motion": False, "motion_score": 0.0, "changed_fraction": 0.0})
        self.previous_frame = current
        return result

    process = perceive
