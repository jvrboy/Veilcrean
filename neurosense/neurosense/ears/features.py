"""Feature extraction for one-dimensional audio signals."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable

import numpy as np


@dataclass(frozen=True)
class AudioFeatures:
    rms: float
    peak: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_bandwidth: float
    dominant_frequency: float
    duration: float
    sample_rate: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def normalize_audio(samples: Any) -> np.ndarray:
    """Return mono audio normalized to ``[-1, 1]``."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("audio must contain at least one sample")
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    arr = np.nan_to_num(arr, copy=False)
    peak = float(np.max(np.abs(arr)))
    if peak <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return arr / peak


def frame_audio(samples: Any, frame_size: int, hop_size: int | None = None) -> np.ndarray:
    """Slice audio into overlapping frames."""
    audio = normalize_audio(samples)
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    hop_size = frame_size if hop_size is None else hop_size
    if hop_size <= 0:
        raise ValueError("hop_size must be positive")
    if len(audio) < frame_size:
        padded = np.zeros(frame_size, dtype=float)
        padded[: len(audio)] = audio
        return padded[None, :]
    starts = range(0, len(audio) - frame_size + 1, hop_size)
    return np.stack([audio[start : start + frame_size] for start in starts], axis=0)


def extract_audio_features(samples: Any, sample_rate: int = 16_000) -> AudioFeatures:
    """Extract amplitude, zero-crossing, and spectral features."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    audio = normalize_audio(samples)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak = float(np.max(np.abs(audio)))
    signs = np.signbit(audio)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if len(audio) > 1 else 0.0
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    total = float(spectrum.sum())
    if total <= 1e-12:
        centroid = bandwidth = dominant = 0.0
    else:
        centroid = float((freqs * spectrum).sum() / total)
        bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * spectrum).sum() / total))
        dominant = float(freqs[int(np.argmax(spectrum))])
    return AudioFeatures(
        rms=rms,
        peak=peak,
        zero_crossing_rate=zcr,
        spectral_centroid=centroid,
        spectral_bandwidth=bandwidth,
        dominant_frequency=dominant,
        duration=float(len(audio) / sample_rate),
        sample_rate=int(sample_rate),
    )


class EarFeatureExtractor:
    """State-less audio feature extractor class."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate

    def transform(self, samples: Any) -> Dict[str, float]:
        return extract_audio_features(samples, self.sample_rate).as_dict()

    def batch_transform(self, signals: Iterable[Any]) -> list[Dict[str, float]]:
        return [self.transform(signal) for signal in signals]
