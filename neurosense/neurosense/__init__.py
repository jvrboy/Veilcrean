"""
NeuroSense — An original, self-contained cognitive architecture for Python.

Eyes (vision), Ears (audio), Neurons (from-scratch neural networks),
Brain (memory, attention, orchestration), Knowledge (graph + inference),
Learning (reinforcement + unsupervised), Language (statistical NLP).

No AI API providers. No pretrained models. No emotions.
Everything is computed locally from first principles.

Quick start:

    from neurosense import Brain

    brain = Brain(name="atlas")
    brain.learn_fact("water", "is_a", "liquid")
    brain.learn_fact("liquid", "can", "flow")
    print(brain.reason("water", "can"))   # -> ['flow'] (inherited via inference)

    # Vision
    percept = brain.see(image_array)      # numpy HxW or HxWx3 array
    # Hearing
    percept = brain.hear(samples, rate)   # 1-D numpy array of audio samples

Author: generated as an original work. License: MIT.
"""

__version__ = "1.0.0"

from .brain.brain import Brain
from .eyes.vision import Eye
from .ears.audio import Ear
from .neurons.network import NeuralNetwork
from .knowledge.graph import KnowledgeGraph
from .knowledge.inference import InferenceEngine
from .learning.reinforcement import QLearner
from .learning.unsupervised import KMeans, SelfOrganizingMap
from .language.text import LanguageCortex

__all__ = [
    "Brain",
    "Eye",
    "Ear",
    "NeuralNetwork",
    "KnowledgeGraph",
    "InferenceEngine",
    "QLearner",
    "KMeans",
    "SelfOrganizingMap",
    "LanguageCortex",
    "__version__",
]
