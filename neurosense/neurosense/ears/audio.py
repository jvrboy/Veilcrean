"""The Ear — high-level auditory perception organ.

Converts raw audio samples into structured AudioPercepts and can learn
to recognize sounds it has heard before.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field

import numpy as np

from .features import (
    audio_signature,
    detect_onsets,
    detect_pitch,
    spectral_centroid,
    zero_crossing_rate,
)

# Note names for pitch -> musical note conversion
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file using only the standard library. Returns (samples, rate).

    Samples are mono float64 in [-1, 1].
    """
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    data /= float(np.iinfo(dtype).max)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, rate


def pitch_to_note(freq: float) -> str | None:
    """Convert a frequency in Hz to the nearest musical note name."""
    if freq <= 0:
        return None
    midi = 69 + 12 * np.log2(freq / 440.0)
    midi_round = int(round(midi))
    return f"{_NOTE_NAMES[midi_round % 12]}{midi_round // 12 - 1}"


@dataclass
class AudioPercept:
    """A structured description of what the ear heard."""

    signature: np.ndarray
    duration: float
    loudness: float
    pitch: float
    note: str | None
    brightness: float
    noisiness: float
    onsets: list = field(default_factory=list)
    label: str | None = None
    confidence: float = 0.0

    def describe(self) -> str:
        parts = []
        parts.append("loud" if self.loudness > 0.3
                     else "quiet" if self.loudness < 0.05 else "moderate")
        if self.pitch > 0:
            parts.append(f"pitched sound at {self.pitch:.0f} Hz"
                         + (f" (~note {self.note})" if self.note else ""))
        elif self.noisiness > 0.3:
            parts.append("noise-like sound")
        else:
            parts.append("unpitched sound")
        parts.append(f"{len(self.onsets)} distinct event(s)")
        parts.append(f"{self.duration:.2f}s long")
        if self.label:
            parts.append(f"recognized as '{self.label}' "
                         f"({self.confidence:.0%} confidence)")
        return "I hear a " + ", ".join(parts) + "."


class Ear:
    """Auditory perception with recognition memory.

    >>> ear = Ear()
    >>> ear.memorize(bell_samples, 44100, "bell")
    >>> percept = ear.perceive(new_samples, 44100)
    >>> percept.label   # 'bell' if it sounds similar
    """

    def __init__(self, recognition_threshold: float = 0.88):
        self.recognition_threshold = recognition_threshold
        self._memory: list[tuple[np.ndarray, str]] = []

    # ------------------------------------------------------------------ #
    def perceive(self, samples: np.ndarray, rate: int) -> AudioPercept:
        """Listen to a signal and produce a full structured percept."""
        samples = np.asarray(samples, dtype=np.float64)
        pitch = detect_pitch(samples, rate)
        signature = audio_signature(samples, rate)
        percept = AudioPercept(
            signature=signature,
            duration=len(samples) / rate,
            loudness=float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0,
            pitch=pitch,
            note=pitch_to_note(pitch),
            brightness=spectral_centroid(samples, rate),
            noisiness=zero_crossing_rate(samples),
            onsets=detect_onsets(samples, rate),
        )
        label, conf = self._recognize(signature)
        if label is not None:
            percept.label = label
            percept.confidence = conf
        return percept

    def perceive_file(self, path: str) -> AudioPercept:
        samples, rate = load_wav(path)
        return self.perceive(samples, rate)

    # ------------------------------------------------------------------ #
    def memorize(self, samples: np.ndarray, rate: int, label: str) -> None:
        """Associate a sound with a label — one-shot auditory learning."""
        self._memory.append((audio_signature(samples, rate), label))

    def _recognize(self, signature: np.ndarray) -> tuple[str | None, float]:
        best_label, best_sim = None, 0.0
        for stored, label in self._memory:
            na, nb = np.linalg.norm(signature), np.linalg.norm(stored)
            sim = float(signature @ stored / (na * nb)) if na and nb else 0.0
            if sim > best_sim:
                best_label, best_sim = label, sim
        if best_sim >= self.recognition_threshold:
            return best_label, best_sim
        return None, best_sim

    def known_labels(self) -> list[str]:
        return sorted({label for _, label in self._memory})
