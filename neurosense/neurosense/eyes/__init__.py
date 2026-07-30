"""Visual sensing helpers."""
from .features import VisualFeatures, EyeFeatureExtractor, edge_magnitude, extract_visual_features, normalize_image, to_grayscale
from .vision import VisionSensor, detect_motion, summarize_scene

__all__ = [
    "VisualFeatures",
    "EyeFeatureExtractor",
    "to_grayscale",
    "normalize_image",
    "edge_magnitude",
    "extract_visual_features",
    "VisionSensor",
    "detect_motion",
    "summarize_scene",
]
