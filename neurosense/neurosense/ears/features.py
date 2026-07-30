"""Low-level auditory feature extraction — the cochlea.

FFT-based spectral analysis, pitch detection via autocorrelation,
onset detection, and compact audio signatures. numpy only.
"""

from __future__ import annotations

import numpy as np


def spectrum(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, magnitudes) of the signal's frequency content."""
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    windowed = samples * np.hanning(n)
    mags = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / rate)
    return freqs, mags


def spectrogram(samples: np.ndarray, rate: int, frame_size: int = 1024,
                hop: int = 512) -> np.ndarray:
    """Short-time Fourier transform magnitude matrix (frames x bins)."""
    samples = np.asarray(samples, dtype=np.float64)
    window = np.hanning(frame_size)
    frames = []
    for start in range(0, len(samples) - frame_size + 1, hop):
        frame = samples[start:start + frame_size] * window
        frames.append(np.abs(np.fft.rfft(frame)))
    if not frames:
        return np.zeros((0, frame_size // 2 + 1))
    return np.array(frames)


def spectral_centroid(samples: np.ndarray, rate: int) -> float:
    """The 'center of mass' of the spectrum — perceptual brightness in Hz."""
    freqs, mags = spectrum(samples, rate)
    total = mags.sum()
    if total == 0:
        return 0.0
    return float((freqs * mags).sum() / total)


def zero_crossing_rate(samples: np.ndarray) -> float:
    """Fraction of sign changes — high for noise/fricatives, low for tones."""
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(samples).astype(int)))))


def detect_pitch(samples: np.ndarray, rate: int,
                 fmin: float = 50.0, fmax: float = 2000.0) -> float:
    """Fundamental frequency estimation via normalized autocorrelation.

    Returns pitch in Hz, or 0.0 if no clear pitch is found.
    """
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples - samples.mean()
    if len(samples) < int(rate / fmin) or not samples.any():
        return 0.0
    corr = np.correlate(samples, samples, mode="full")[len(samples) - 1:]
    if corr[0] <= 0:
        return 0.0
    corr = corr / corr[0]
    lag_min = max(1, int(rate / fmax))
    lag_max = min(len(corr) - 1, int(rate / fmin))
    if lag_max <= lag_min:
        return 0.0
    segment = corr[lag_min:lag_max]
    peak = int(np.argmax(segment)) + lag_min
    if corr[peak] < 0.3:  # weak periodicity => unvoiced / noise
        return 0.0
    return float(rate / peak)


def detect_onsets(samples: np.ndarray, rate: int, frame_size: int = 1024,
                  hop: int = 512, sensitivity: float = 1.5) -> list[float]:
    """Detect event start times (seconds) via spectral-flux peak picking."""
    spec = spectrogram(samples, rate, frame_size, hop)
    if len(spec) < 3:
        return []
    flux = np.maximum(np.diff(spec, axis=0), 0).sum(axis=1)
    threshold = flux.mean() + sensitivity * flux.std()
    onsets = []
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            onsets.append(float((i + 1) * hop / rate))
    return onsets


def mel_filterbank(n_filters: int, n_fft_bins: int, rate: int) -> np.ndarray:
    """Triangular mel-scale filterbank matrix (n_filters x n_fft_bins)."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(rate / 2), n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft_bins - 1) * 2 * hz_points / rate).astype(int)
    bins = np.clip(bins, 0, n_fft_bins - 1)
    bank = np.zeros((n_filters, n_fft_bins))
    for i in range(n_filters):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, center):
            if center > left:
                bank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                bank[i, j] = (right - j) / (right - center)
    return bank


def audio_signature(samples: np.ndarray, rate: int,
                    n_filters: int = 20) -> np.ndarray:
    """Compact, comparable feature vector for any sound.

    Mel-band energies + centroid + ZCR + pitch + loudness.
    Length = n_filters + 4.
    """
    samples = np.asarray(samples, dtype=np.float64)
    spec = spectrogram(samples, rate)
    if len(spec) == 0:
        return np.zeros(n_filters + 4)
    bank = mel_filterbank(n_filters, spec.shape[1], rate)
    mel_energy = np.log1p(spec @ bank.T).mean(axis=0)
    norm = np.linalg.norm(mel_energy)
    if norm > 0:
        mel_energy = mel_energy / norm
    return np.concatenate([
        mel_energy,
        [
            spectral_centroid(samples, rate) / (rate / 2),
            zero_crossing_rate(samples),
            detect_pitch(samples, rate) / 2000.0,
            float(np.sqrt(np.mean(samples**2))),
        ],
    ])
