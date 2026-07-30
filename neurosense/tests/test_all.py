from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def test_public_imports():
    import neurosense
    from neurosense import NeuroSenseAgent, NeuralNetwork
    from neurosense.eyes import extract_visual_features
    from neurosense.ears import extract_audio_features
    from neurosense.knowledge import KnowledgeGraph, InferenceEngine

    assert neurosense.__version__
    assert NeuroSenseAgent is not None
    assert NeuralNetwork is not None
    assert extract_visual_features(np.zeros((4, 4))).width == 4
    assert extract_audio_features(np.zeros(8)).rms == 0.0
    graph = KnowledgeGraph()
    graph.add_edge("cat", "is_a", "animal")
    graph.add_edge("animal", "is_a", "organism")
    inferred = InferenceEngine(graph).infer()
    assert any(t.subject == "cat" and t.object == "organism" for t in inferred)


def test_visual_and_audio_features():
    from neurosense.eyes import VisionSensor, detect_motion, extract_visual_features
    from neurosense.ears import AudioSensor, tone

    image = np.eye(8)
    vf = extract_visual_features(image)
    assert 0.0 <= vf.brightness <= 1.0
    assert vf.width == 8 and vf.height == 8
    motion = detect_motion(np.zeros((8, 8)), image)
    assert motion["motion_score"] > 0
    assert VisionSensor().perceive(image)["modality"] == "vision"

    sig = tone(440, duration=0.05, sample_rate=8000)
    audio = AudioSensor(sample_rate=8000).perceive(sig)
    assert audio["modality"] == "audio"
    assert 400 <= audio["features"]["dominant_frequency"] <= 480


def test_neural_network_and_learning():
    from neurosense.neurons import DenseLayer, NeuralNetwork, relu, sigmoid
    from neurosense.learning import KMeans, PCA, QLearningAgent

    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    net = NeuralNetwork([DenseLayer(2, 3, activation="relu", seed=1), DenseLayer(3, 1, activation="sigmoid", seed=2)])
    y = net.predict(x)
    assert y.shape == (2, 1)
    loss = net.train_step(x, np.array([[0.0], [1.0]]), lr=0.01)
    assert loss >= 0

    data = np.array([[0, 0], [0.1, 0.2], [4, 4], [4.1, 3.9]], dtype=float)
    labels = KMeans(n_clusters=2, seed=4).fit_predict(data)
    assert set(labels) == {0, 1}
    assert PCA(n_components=1).fit_transform(data).shape == (4, 1)

    agent = QLearningAgent(actions=["a", "b"], epsilon=0.0)
    agent.update("s", "b", 1.0, "terminal", done=True)
    assert agent.act("s") == "b"


def test_language_memory_and_agent():
    from neurosense import NeuroSenseAgent
    from neurosense.brain import MemoryStore
    from neurosense.language import TextSensor, TextVectorizer, sentiment, tokenize

    assert tokenize("Hello, NeuroSense!") == ["hello", "neurosense"]
    assert sentiment("good calm clear") > 0
    vec = TextVectorizer().fit_transform(["calm focus", "risk error"])
    assert vec.shape[0] == 2
    assert TextSensor().perceive("calm focused learning")["features"]["token_count"] == 3

    mem = MemoryStore()
    mem.add("calm focused memory", tags={"note"})
    assert mem.recall("focused", tag="note")

    agent = NeuroSenseAgent()
    thought = agent.step({"text": "calm focused learning", "image": np.zeros((6, 6))})
    assert thought.focus
    assert thought.confidence >= 0
    assert agent.act(thought)["action"] in {"observe", "respond"}
