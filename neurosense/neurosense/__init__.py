"""NeuroSense public API.

The package is intentionally small: every major subsystem is implemented with
NumPy and standard-library building blocks so the same code can run in tests,
notebooks, and lightweight agents.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .brain.brain import Brain, NeuroSenseAgent, SensorySignal, Thought
from .brain.attention import AttentionMechanism, FocusState
from .brain.memory import MemoryItem, MemoryStore
from .knowledge.graph import KnowledgeGraph, Triple
from .knowledge.inference import InferenceEngine
from .neurons.network import NeuralNetwork
from .neurons.layers import DenseLayer, DropoutLayer, LayerNorm
from .neurons.activations import get_activation

__version__ = "0.1.0"


def seed(value: Optional[int] = None) -> int:
    """Seed Python and NumPy RNGs, returning the seed used."""
    if value is None:
        value = random.randrange(0, 2**32 - 1)
    random.seed(value)
    np.random.seed(value)
    return value


__all__ = [
    "__version__",
    "seed",
    "Brain",
    "NeuroSenseAgent",
    "SensorySignal",
    "Thought",
    "AttentionMechanism",
    "FocusState",
    "MemoryItem",
    "MemoryStore",
    "KnowledgeGraph",
    "Triple",
    "InferenceEngine",
    "NeuralNetwork",
    "DenseLayer",
    "DropoutLayer",
    "LayerNorm",
    "get_activation",
]
