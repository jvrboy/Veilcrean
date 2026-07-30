"""Low-level visual feature extraction — the retina and V1 cortex.

Every operation is implemented from first principles with numpy.
No OpenCV, no pretrained models, no external APIs.
"""

from __future__ import annotations

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image (HxW, HxWx3, or HxWx4) to float grayscale in [0, 1]."""
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        rgb = img[..., :3]
        # Luminance weights matching human photoreceptor sensitivity
        return rgb @ np.array([0.2126, 0.7152, 0.0722])
    raise ValueError(f"Unsupported image shape: {img.shape}")


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution with reflect padding, implemented via stride tricks."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel[::-1, ::-1])


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax**2) / (2 * sigma**2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()


def gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    return convolve2d(image, gaussian_kernel(size, sigma))


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def sobel_edges(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (edge magnitude, edge orientation in radians)."""
    gx = convolve2d(gray, SOBEL_X)
    gy = convolve2d(gray, SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)
    return magnitude, orientation


def image_histogram(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    """Normalized intensity histogram — a global brightness fingerprint."""
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    return hist / total if total > 0 else hist.astype(np.float64)


def detect_corners(gray: np.ndarray, k: float = 0.05, threshold: float = 0.01,
                   max_corners: int = 200) -> list[tuple[int, int]]:
    """Harris corner detection. Returns list of (row, col) corner points."""
    gx = convolve2d(gray, SOBEL_X)
    gy = convolve2d(gray, SOBEL_Y)
    ixx = gaussian_blur(gx * gx, 5, 1.5)
    iyy = gaussian_blur(gy * gy, 5, 1.5)
    ixy = gaussian_blur(gx * gy, 5, 1.5)
    det = ixx * iyy - ixy**2
    trace = ixx + iyy
    response = det - k * trace**2
    peak = response.max()
    if peak <= 0:
        return []
    mask = response > threshold * peak
    coords = np.argwhere(mask)
    strengths = response[mask]
    order = np.argsort(strengths)[::-1][:max_corners]
    return [tuple(int(v) for v in coords[i]) for i in order]


def find_blobs(gray: np.ndarray, threshold: float = 0.5,
               min_area: int = 20) -> list[dict]:
    """Connected-component blob detection via iterative flood fill.

    Returns a list of blobs: {"centroid": (r, c), "area": int, "bbox": (r0, c0, r1, c1)}.
    """
    binary = gray > threshold
    visited = np.zeros_like(binary, dtype=bool)
    h, w = binary.shape
    blobs = []
    for sr in range(h):
        for sc in range(w):
            if not binary[sr, sc] or visited[sr, sc]:
                continue
            # Iterative flood fill (stack-based, no recursion limits)
            stack = [(sr, sc)]
            visited[sr, sc] = True
            pixels = []
            while stack:
                r, c = stack.pop()
                pixels.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if len(pixels) >= min_area:
                arr = np.array(pixels)
                blobs.append({
                    "centroid": (float(arr[:, 0].mean()), float(arr[:, 1].mean())),
                    "area": len(pixels),
                    "bbox": (int(arr[:, 0].min()), int(arr[:, 1].min()),
                             int(arr[:, 0].max()), int(arr[:, 1].max())),
                })
    blobs.sort(key=lambda b: -b["area"])
    return blobs


def orientation_histogram(magnitude: np.ndarray, orientation: np.ndarray,
                          bins: int = 12) -> np.ndarray:
    """Edge-orientation histogram weighted by edge strength (HOG-like global)."""
    hist, _ = np.histogram(orientation, bins=bins, range=(-np.pi, np.pi),
                           weights=magnitude)
    total = hist.sum()
    return hist / total if total > 0 else hist


def image_signature(image: np.ndarray, grid: int = 4) -> np.ndarray:
    """A compact, comparable feature vector for any image.

    Combines: coarse intensity grid, brightness histogram, and
    edge-orientation histogram. Suitable for similarity search and
    as neural-network input. Length = grid*grid + 32 + 12.
    """
    gray = to_grayscale(image)
    h, w = gray.shape
    # Coarse spatial grid of mean intensities
    cells = []
    for i in range(grid):
        for j in range(grid):
            cell = gray[i * h // grid:(i + 1) * h // grid,
                        j * w // grid:(j + 1) * w // grid]
            cells.append(cell.mean() if cell.size else 0.0)
    mag, ori = sobel_edges(gray)
    return np.concatenate([
        np.array(cells),
        image_histogram(gray, 32),
        orientation_histogram(mag, ori, 12),
    ])
