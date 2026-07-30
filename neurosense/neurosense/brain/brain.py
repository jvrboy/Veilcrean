"""Brain — the central cognitive agent.

Wires together every subsystem:

    Eyes  -> visual percepts    -> attention -> memory
    Ears  -> auditory percepts  -> attention -> memory
    Text  -> language cortex    -> fact extraction -> knowledge graph
    Knowledge graph <-> inference engine (reasoning)
    Working memory (current thoughts) + episodic memory (life story)
    Q-learning (learning from consequences)
    Neural networks (trainable pattern classifiers)

No emotions. No API calls. Everything it knows, it learned.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..eyes.vision import Eye, VisualPercept
from ..ears.audio import Ear, AudioPercept
from ..knowledge.graph import KnowledgeGraph, Fact
from ..knowledge.inference import InferenceEngine
from ..language.text import LanguageCortex
from ..learning.reinforcement import QLearner
from ..neurons.network import NeuralNetwork
from .attention import Attention
from .memory import WorkingMemory, EpisodicMemory


class Brain:
    """A complete cognitive agent with senses, memory, knowledge, and learning.

    >>> brain = Brain(name="atlas")
    >>> brain.read("A dog is a mammal. Mammals have fur. A dog can bark.")
    >>> brain.reason("dog", "has")
    ['fur']
    >>> print(brain.introspect())
    """

    def __init__(self, name: str = "brain"):
        self.name = name
        # Senses
        self.eye = Eye()
        self.ear = Ear()
        # Cognition
        self.attention = Attention()
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.knowledge = KnowledgeGraph()
        self.inference = InferenceEngine(self.knowledge)
        self.language = LanguageCortex()
        # Learning
        self._classifiers: dict[str, NeuralNetwork] = {}
        self._agents: dict[str, QLearner] = {}

    # ============================ PERCEPTION =========================== #
    def see(self, image: np.ndarray, label: str | None = None) -> VisualPercept:
        """Look at an image. Optionally teach it what this is (label)."""
        percept = self.eye.perceive(image)
        salience = self.attention.evaluate(
            percept.signature, intensity=percept.contrast, channel="vision")
        if label:
            self.eye.memorize(image, label)
            self.knowledge.add(label, "is_a", "visual object")
            percept.label = label
            percept.confidence = 1.0
        focus = percept.label or "unlabeled sight"
        self.working_memory.hold(f"sight:{focus}", activation=salience)
        self.episodic_memory.record(
            "sight", percept.describe(), importance=salience,
            data={"label": percept.label})
        return percept

    def hear(self, samples: np.ndarray, rate: int,
             label: str | None = None) -> AudioPercept:
        """Listen to audio samples. Optionally teach it what this is."""
        percept = self.ear.perceive(samples, rate)
        salience = self.attention.evaluate(
            percept.signature, intensity=percept.loudness, channel="hearing")
        if label:
            self.ear.memorize(samples, rate, label)
            self.knowledge.add(label, "is_a", "sound")
            percept.label = label
            percept.confidence = 1.0
        focus = percept.label or "unlabeled sound"
        self.working_memory.hold(f"sound:{focus}", activation=salience)
        self.episodic_memory.record(
            "sound", percept.describe(), importance=salience,
            data={"label": percept.label})
        return percept

    # ============================ KNOWLEDGE ============================ #
    def read(self, text: str) -> list[Fact]:
        """Read text: absorb language statistics AND extract facts into
        the knowledge graph. Returns the facts learned."""
        self.language.read(text)
        learned = []
        for s, r, o in self.language.extract_facts(text):
            fact = self.knowledge.add(s, r, o, confidence=0.8)
            learned.append(fact)
            self.working_memory.hold(f"fact:{s} {r} {o}", activation=0.6)
            self.episodic_memory.record("fact", str(fact), importance=0.55)
        if learned:
            self.episodic_memory.record(
                "reading", f"Read text and learned {len(learned)} fact(s).",
                importance=min(1.0, 0.3 + 0.1 * len(learned)))
        return learned

    def learn_fact(self, subject: str, relation: str, obj: str,
                   confidence: float = 1.0) -> Fact:
        """Directly teach the brain a fact."""
        fact = self.knowledge.add(subject, relation, obj, confidence)
        self.working_memory.hold(f"fact:{subject} {relation} {obj}")
        self.episodic_memory.record("fact", str(fact), importance=0.6)
        return fact

    def reason(self, subject: str, relation: str) -> list[str]:
        """What does `subject` `relation`? Runs inference first, so it
        returns both stored and logically derived answers."""
        self.inference.infer()
        return [f.obj for f in self.knowledge.query(subject=subject,
                                                    relation=relation)]

    def ask(self, subject: str, relation: str, obj: str) -> tuple[bool, float, str]:
        """Yes/no question with confidence and explanation."""
        answer, conf, _ = self.inference.ask(subject, relation, obj)
        why = self.inference.why(subject, obj) if answer else \
            f"I have no knowledge that {subject} {relation} {obj}."
        return answer, conf, why

    def free_associate(self, concept: str, top: int = 8) -> list[str]:
        """Think loosely around a concept (spreading activation +
        learned word co-occurrence)."""
        graph_assoc = list(self.knowledge.spread_activation(concept))[:top]
        word_assoc = [w for w, _ in self.language.associate(concept, top)]
        merged = []
        for item in graph_assoc + word_assoc:
            if item not in merged and item != concept:
                merged.append(item)
        return merged[:top]

    # ============================ LEARNING ============================= #
    def build_classifier(self, name: str, input_size: int,
                         classes: list[str],
                         hidden: int = 32) -> NeuralNetwork:
        """Grow a dedicated neural circuit for a classification skill."""
        net = NeuralNetwork([input_size, hidden, len(classes)],
                            activation="relu", output="softmax")
        net.class_names = classes  # type: ignore[attr-defined]
        self._classifiers[name] = net
        return net

    def train_classifier(self, name: str, X: np.ndarray, labels: list[str],
                         epochs: int = 200) -> list[float]:
        net = self._classifiers[name]
        classes = net.class_names  # type: ignore[attr-defined]
        Y = np.zeros((len(labels), len(classes)))
        for i, lab in enumerate(labels):
            Y[i, classes.index(lab)] = 1.0
        history = net.train(np.asarray(X), Y, epochs=epochs)
        self.episodic_memory.record(
            "training", f"Trained skill '{name}' to loss {history[-1]:.4f}.",
            importance=0.7)
        return history

    def classify(self, name: str, x: np.ndarray) -> tuple[str, float]:
        net = self._classifiers[name]
        probs = net.predict(x)[0]
        idx = int(np.argmax(probs))
        return net.class_names[idx], float(probs[idx])  # type: ignore[attr-defined]

    def get_agent(self, name: str, actions: list) -> QLearner:
        """A named reinforcement-learning agent (created on first use)."""
        if name not in self._agents:
            self._agents[name] = QLearner(actions)
        return self._agents[name]

    # ============================ COGNITION ============================ #
    def think(self) -> str:
        """One cognitive cycle: consolidate, infer, report current focus."""
        derived = self.inference.infer()
        self.working_memory.tick()
        focus = self.working_memory.focus()
        lines = []
        if focus:
            lines.append(f"Current focus: {focus}.")
        if derived:
            lines.append(f"Derived {len(derived)} new conclusion(s), "
                         f"e.g. {derived[0]}.")
        if not lines:
            lines.append("Mind is quiet; awaiting input.")
        return " ".join(lines)

    def sleep(self) -> str:
        """Consolidate memories and strengthen knowledge (like real sleep)."""
        forgotten = self.episodic_memory.consolidate()
        derived = self.inference.infer()
        self.attention.reset_habituation()
        return (f"Slept. Forgot {forgotten} trivial memories, "
                f"derived {len(derived)} new facts, attention refreshed.")

    def introspect(self) -> str:
        """A self-report of the brain's current state."""
        wm = ", ".join(f"{k} ({v:.2f})"
                       for k, v in self.working_memory.contents()) or "empty"
        return "\n".join([
            f"=== {self.name} ===",
            f"Knowledge:        {len(self.knowledge)} facts, "
            f"{len(self.knowledge.entities())} concepts",
            f"Episodic memory:  {len(self.episodic_memory)} experiences",
            f"Working memory:   {wm}",
            f"Vision memory:    {len(self.eye.known_labels())} known objects "
            f"{self.eye.known_labels()}",
            f"Audio memory:     {len(self.ear.known_labels())} known sounds "
            f"{self.ear.known_labels()}",
            f"Vocabulary:       {len(self.language.vocabulary)} words",
            f"Trained skills:   {list(self._classifiers)}",
            f"RL agents:        {list(self._agents)}",
        ])

    # ============================ PERSISTENCE ========================== #
    def save(self, directory: str) -> None:
        """Persist knowledge, memories, and language stats to a directory."""
        os.makedirs(directory, exist_ok=True)
        self.knowledge.save(os.path.join(directory, "knowledge.json"))
        self.episodic_memory.save(os.path.join(directory, "episodes.json"))
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"name": self.name}, f)
        for name, net in self._classifiers.items():
            net.save(os.path.join(directory, f"classifier_{name}.json"))

    @classmethod
    def load(cls, directory: str) -> "Brain":
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        brain = cls(name=meta["name"])
        kg_path = os.path.join(directory, "knowledge.json")
        if os.path.exists(kg_path):
            brain.knowledge = KnowledgeGraph.load(kg_path)
            brain.inference = InferenceEngine(brain.knowledge)
        ep_path = os.path.join(directory, "episodes.json")
        if os.path.exists(ep_path):
            brain.episodic_memory = EpisodicMemory.load(ep_path)
        return brain
