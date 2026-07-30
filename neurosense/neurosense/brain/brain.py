"""An agent that combines senses, memory, attention, and knowledge."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np

from ..ears.audio import AudioSensor
from ..eyes.vision import VisionSensor
from ..knowledge.graph import KnowledgeGraph
from ..language.text import TextSensor
from .attention import AttentionMechanism
from .memory import MemoryStore


@dataclass(frozen=True)
class SensorySignal:
    modality: str
    summary: str
    features: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass(frozen=True)
class Thought:
    summary: str
    focus: List[SensorySignal]
    memories: list[Any] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class NeuroSenseAgent:
    """Small multimodal agent with deterministic behavior."""

    memory: MemoryStore = field(default_factory=MemoryStore)
    attention: AttentionMechanism = field(default_factory=AttentionMechanism)
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    vision: VisionSensor = field(default_factory=VisionSensor)
    audio: AudioSensor = field(default_factory=AudioSensor)
    text: TextSensor = field(default_factory=TextSensor)

    def perceive(self, inputs: Dict[str, Any] | Iterable[SensorySignal]) -> list[SensorySignal]:
        if not isinstance(inputs, dict):
            return list(inputs)
        signals: list[SensorySignal] = []
        if "image" in inputs:
            perceived = self.vision.perceive(inputs["image"])
            signals.append(SensorySignal("vision", perceived["summary"], perceived["features"], inputs["image"]))
        if "audio" in inputs:
            perceived = self.audio.perceive(inputs["audio"])
            signals.append(SensorySignal("audio", perceived["summary"], perceived["features"], inputs["audio"]))
        if "text" in inputs:
            perceived = self.text.perceive(inputs["text"])
            signals.append(SensorySignal("language", perceived["summary"], perceived["features"], inputs["text"]))
        for key, value in inputs.items():
            if key not in {"image", "audio", "text"}:
                signals.append(SensorySignal(key, str(value), {}, value))
        return signals

    def think(self, signals: Iterable[SensorySignal]) -> Thought:
        signals = list(signals)
        focus_state = self.attention.focus([{"features": sig.features} for sig in signals])
        focused = [signals[idx] for idx in focus_state.indices]
        query = " ".join(sig.summary for sig in focused) if focused else ""
        memories = [item.content for item in self.memory.recall(query, k=3)] if query else []
        summary = "; ".join(sig.summary for sig in focused) if focused else "no salient input"
        confidence = min(1.0, focus_state.salience / (focus_state.salience + 1.0)) if focus_state.salience >= 0 else 0.0
        return Thought(summary=summary, focus=focused, memories=memories, confidence=float(confidence))

    def learn(self, thought: Thought) -> None:
        self.memory.add(thought.summary, tags={"thought"}, strength=max(0.1, thought.confidence))
        for signal in thought.focus:
            self.memory.add(signal.summary, tags={signal.modality}, strength=0.5 + thought.confidence)
            self.graph.add_node(signal.modality)
            self.graph.add_node(signal.summary)
            self.graph.add_edge(signal.modality, "observes", signal.summary)

    def act(self, thought: Thought) -> Dict[str, Any]:
        return {
            "action": "observe" if thought.confidence < 0.4 else "respond",
            "summary": thought.summary,
            "confidence": thought.confidence,
        }

    def step(self, inputs: Dict[str, Any] | Iterable[SensorySignal]) -> Thought:
        signals = self.perceive(inputs)
        thought = self.think(signals)
        self.learn(thought)
        return thought


Brain = NeuroSenseAgent
