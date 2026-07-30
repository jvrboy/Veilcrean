"""Audio utilities and high-level sensor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .features import extract_audio_features, normalize_audio


def tone(frequency: float = 440.0, duration: float = 1.0, sample_rate: int = 16_000, amplitude: float = 1.0) -> np.ndarray:
    """Generate a sine tone."""
    if duration <= 0 or sample_rate <= 0:
        raise ValueError("duration and sample_rate must be positive")
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def mix(*signals: Any, normalize: bool = True) -> np.ndarray:
    """Mix signals by zero-padding to the longest length."""
    if not signals:
        return np.array([], dtype=float)
    arrays = [np.asarray(sig, dtype=float).reshape(-1) for sig in signals]
    n = max(len(arr) for arr in arrays)
    out = np.zeros(n, dtype=float)
    for arr in arrays:
        out[: len(arr)] += arr
    return normalize_audio(out) if normalize and n else out


@dataclass
class AudioSensor:
    sample_rate: int = 16_000

    def perceive(self, samples: Any) -> Dict[str, Any]:
        features = extract_audio_features(samples, self.sample_rate)
        loudness = "loud" if features.rms > 0.5 else "quiet" if features.rms < 0.1 else "moderate"
        pitch = "low" if features.dominant_frequency < 250 else "high" if features.dominant_frequency > 2000 else "mid"
        return {
            "modality": "audio",
            "features": features.as_dict(),
            "summary": f"{loudness} {pitch}-frequency sound",
        }

    process = perceive
