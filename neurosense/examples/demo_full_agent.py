"""Full cognitive agent demo: eyes + ears + brain + reasoning + learning.

Run:  python examples/demo_full_agent.py
"""

import os
import sys

# Make the package importable when running from the examples/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from neurosense import Brain


def make_circle(size=64):
    y, x = np.mgrid[0:size, 0:size]
    return (((x - size / 2) ** 2 + (y - size / 2) ** 2)
            < (size / 3) ** 2).astype(float)


def make_square(size=64):
    img = np.zeros((size, size))
    img[size // 4:3 * size // 4, size // 4:3 * size // 4] = 1.0
    return img


def make_tone(freq, rate=16000, duration=0.5):
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def main():
    brain = Brain(name="atlas")

    # ---- EYES: one-shot visual learning + recognition ----
    print("--- VISION ---")
    brain.see(make_circle(), label="circle")
    brain.see(make_square(), label="square")
    percept = brain.see(make_circle())  # unseen instance
    print(percept.describe())

    # ---- EARS: sound learning + recognition + pitch ----
    print("\n--- HEARING ---")
    brain.hear(make_tone(440), 16000, label="tuning fork A")
    percept = brain.hear(make_tone(440), 16000)
    print(percept.describe())

    # ---- READING: learn facts from natural language ----
    print("\n--- READING & REASONING ---")
    brain.read(
        "A dog is a mammal. A mammal is an animal. Mammals have fur. "
        "An animal can move. Dogs can bark. A cat is a mammal. "
        "Cats can climb. The whale is a mammal. Whales live in the ocean."
    )
    print("What does a dog have?  ->", brain.reason("dog", "has"))
    print("What can a cat do?     ->", brain.reason("cat", "can"))
    ok, conf, why = brain.ask("whale", "has", "fur")
    print(f"Does a whale have fur? -> {ok} ({conf:.0%}). {why}")

    # ---- FREE ASSOCIATION (creative thought) ----
    print("\n--- FREE ASSOCIATION ---")
    print("Thinking about 'dog':", brain.free_associate("dog"))

    # ---- NEURAL SKILL: train a classifier circuit ----
    print("\n--- NEURAL LEARNING ---")
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (50, 4)), rng.normal(3, 1, (50, 4))])
    labels = ["low"] * 50 + ["high"] * 50
    brain.build_classifier("magnitude", input_size=4, classes=["low", "high"])
    brain.train_classifier("magnitude", X, labels, epochs=100)
    pred, prob = brain.classify("magnitude", np.array([3.1, 2.8, 3.3, 2.9]))
    print(f"Classified [3.1 2.8 3.3 2.9] as '{pred}' ({prob:.0%})")

    # ---- COGNITIVE CYCLE + SLEEP ----
    print("\n--- COGNITION ---")
    print(brain.think())
    print(brain.sleep())

    # ---- MEMORY RECALL ----
    print("\n--- EPISODIC RECALL ---")
    for ep in brain.episodic_memory.recall("fact learned", top=3):
        print(" *", ep.summary)

    # ---- INTROSPECTION ----
    print("\n" + brain.introspect())


if __name__ == "__main__":
    main()
