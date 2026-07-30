"""NeuroSense test suite — run with: python tests/test_all.py"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neurosense import (
    Brain, Eye, Ear, NeuralNetwork, KnowledgeGraph, InferenceEngine,
    QLearner, KMeans, SelfOrganizingMap, LanguageCortex,
)
from neurosense.neurons import HebbianLayer, SpikingNetwork
from neurosense.neurons.hebbian import AssociativeMemory

PASSED = 0
FAILED = []


def check(name, condition):
    global PASSED
    if condition:
        PASSED += 1
        print(f"  ok  {name}")
    else:
        FAILED.append(name)
        print(f" FAIL {name}")


def circle(size=48):
    y, x = np.mgrid[0:size, 0:size]
    return (((x - size / 2) ** 2 + (y - size / 2) ** 2) < (size / 3) ** 2).astype(float)


def square(size=48):
    img = np.zeros((size, size))
    img[10:38, 10:38] = 1.0
    return img


def tone(freq, rate=16000, dur=0.4):
    t = np.linspace(0, dur, int(rate * dur), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)


# ------------------------------ eyes ---------------------------------- #
print("EYES")
eye = Eye()
eye.memorize(circle(), "circle")
eye.memorize(square(), "square")
p = eye.perceive(circle())
check("recognizes a circle", p.label == "circle")
check("finds blobs", len(p.blobs) >= 1)
check("describe() speaks", "I see" in p.describe())

# ------------------------------ ears ---------------------------------- #
print("EARS")
ear = Ear()
p = ear.perceive(tone(440), 16000)
check("pitch ~440Hz", abs(p.pitch - 440) < 15)
check("note is A4", p.note == "A4")
ear.memorize(tone(440), 16000, "A tone")
p2 = ear.perceive(tone(442), 16000)
check("recognizes similar tone", p2.label == "A tone")

# ---------------------------- neurons --------------------------------- #
print("NEURONS")
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)
net = NeuralNetwork([2, 8, 2], activation="tanh", seed=1)
net.train(X, Y, epochs=800, batch_size=4)
check("learns XOR", all(net.predict_class(x) == int(y[1])
                        for x, y in zip(X, Y)))
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "net.json")
    net.save(path)
    net2 = NeuralNetwork.load(path)
    check("save/load preserves predictions",
          net2.predict_class(X[1]) == net.predict_class(X[1]))

hebb = HebbianLayer(4, 1, seed=0)
for _ in range(300):
    hebb.observe(np.array([1.0, 1.0, -1.0, -1.0])
                 * np.random.default_rng().normal(1, 0.1))
w = hebb.W[0]
check("Hebbian finds correlation", np.sign(w[0]) == np.sign(w[1]) != np.sign(w[2]))

mem = AssociativeMemory(16)
pat = np.array([1, -1] * 8, dtype=float)
mem.store(pat)
noisy = pat.copy()
noisy[0] *= -1
check("Hopfield recall", np.array_equal(mem.recall(noisy), pat))

snn = SpikingNetwork(20, seed=2)
stim = np.zeros(20)
stim[:4] = 1.5
for _ in range(50):
    snn.step(stim)
check("spiking neurons fire", snn.firing_rates(50)[:4].mean() > 0)

# --------------------------- knowledge -------------------------------- #
print("KNOWLEDGE")
kg = KnowledgeGraph()
kg.add("dog", "is_a", "mammal")
kg.add("mammal", "has", "fur")
kg.add("mammal", "is_a", "animal")
kg.add("animal", "can", "move")
engine = InferenceEngine(kg)
ok, conf, proof = engine.ask("dog", "has", "fur")
check("infers dog has fur", ok and conf > 0.5)
ok, _, _ = engine.ask("dog", "can", "move")
check("chains two levels (dog can move)", ok)
check("path finding", kg.find_path("dog", "animal") is not None)
check("spreading activation", "fur" in kg.spread_activation("dog"))
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "kg.json")
    kg.save(path)
    kg2 = KnowledgeGraph.load(path)
    check("graph save/load", len(kg2) == len(kg))

# ---------------------------- learning -------------------------------- #
print("LEARNING")
agent = QLearner(actions=["a", "b"], seed=0)
for _ in range(200):
    s = "s"
    act = agent.choose(s)
    agent.learn(s, act, 1.0 if act == "a" else -1.0, "s2", done=True)
check("Q-learning prefers rewarded action", agent.best_action("s") == "a")

rng = np.random.default_rng(0)
data = np.vstack([rng.normal(0, .3, (30, 2)), rng.normal(5, .3, (30, 2))])
km = KMeans(k=2, seed=0).fit(data)
check("KMeans separates clusters",
      km.predict(np.array([0, 0])) != km.predict(np.array([5, 5])))

som = SelfOrganizingMap(4, 4, 2, seed=0).fit(data, epochs=10)
check("SOM separates clusters",
      som.locate(np.array([0, 0])) != som.locate(np.array([5, 5])))

# ---------------------------- language -------------------------------- #
print("LANGUAGE")
lc = LanguageCortex(seed=0)
lc.read("The sun is a star. Stars can shine. The sun has heat.")
facts = lc.extract_facts("The sun is a star.")
check("extracts is_a fact", ("sun", "is_a", "star") in facts)
check("generates text", len(lc.generate(seed="the sun")) > 0)
check("similarity search",
      "star" in lc.most_similar("what is the sun", top=1)[0][0].lower())

# ------------------------------ brain --------------------------------- #
print("BRAIN")
brain = Brain(name="test")
brain.see(circle(), label="circle")
p = brain.see(circle())
check("brain sees and recognizes", p.label == "circle")
brain.hear(tone(440), 16000, label="beep")
brain.read("A cat is a mammal. Mammals have fur.")
check("brain reasons with inheritance", "fur" in brain.reason("cat", "has"))
check("free association", len(brain.free_associate("cat")) > 0)
check("working memory holds focus", brain.working_memory.focus() is not None)
check("episodic memory records", len(brain.episodic_memory) >= 3)
check("recall works", len(brain.episodic_memory.recall("fact")) > 0)
check("think() speaks", len(brain.think()) > 0)
check("sleep() consolidates", "Slept" in brain.sleep())
brain.build_classifier("t", input_size=2, classes=["x", "y"])
brain.train_classifier("t", np.array([[0., 0.], [1., 1.]]), ["x", "y"], epochs=200)
check("brain classifier trains", brain.classify("t", np.array([1., 1.]))[0] == "y")
with tempfile.TemporaryDirectory() as d:
    brain.save(d)
    brain2 = Brain.load(d)
    check("brain save/load", "fur" in brain2.reason("cat", "has"))

# ------------------------------ result -------------------------------- #
print(f"\n{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    print("Failures:", FAILED)
    sys.exit(1)
print("ALL TESTS PASSED")
