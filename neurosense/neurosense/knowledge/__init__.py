"""Knowledge — semantic memory: a graph of facts plus logical inference."""

from .graph import KnowledgeGraph, Fact
from .inference import InferenceEngine, Rule

__all__ = ["KnowledgeGraph", "Fact", "InferenceEngine", "Rule"]
