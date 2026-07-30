# NeuroSense

**An original, fully self-contained cognitive architecture for Python.**

Eyes. Ears. Brain. Neurons. Knowledge. Learning. Language.
**No AI API providers. No pretrained models. No emotions.**
Everything is computed locally, from first principles, using only `numpy`.

```
                      +---------------------------+
   image ---> EYES ---|                           |
                      |          BRAIN            |---> reasoning
   audio ---> EARS ---|  attention | working mem  |---> answers
                      |  episodic  | knowledge    |---> decisions
   text  ---> LANG ---|  inference | neurons      |---> generated text
                      +---------------------------+
```

## Install

```bash
pip install numpy          # the only dependency
pip install .              # from this directory
```

## 60-second tour

```python
import numpy as np
from neurosense import Brain

brain = Brain(name="atlas")

# --- EYES: one-shot visual learning ---
brain.see(circle_image, label="circle")      # teach
percept = brain.see(another_circle)          # recognize
print(percept.describe())
# "I see a bright scene, 1 distinct object region(s), ... recognized as 'circle'"

# --- EARS: sound recognition + pitch ---
brain.hear(bell_samples, 44100, label="bell")
percept = brain.hear(new_samples, 44100)
print(percept.pitch, percept.note)           # e.g. 440.0 'A4'

# --- READING: learn facts from plain English ---
brain.read("A dog is a mammal. Mammals have fur. Dogs can bark.")
brain.reason("dog", "has")                   # ['fur']  <- inferred!

# --- ASKING with explanations ---
ok, conf, why = brain.ask("dog", "has", "fur")
# True, 0.72, "Because: dog is_a mammal ; mammal has fur."

# --- CREATIVE free association ---
brain.free_associate("dog")                  # ['mammal', 'fur', 'bark', ...]

# --- NEURAL SKILLS: grow trainable circuits ---
brain.build_classifier("shapes", input_size=60, classes=["circle", "square"])
brain.train_classifier("shapes", X, labels)
brain.classify("shapes", x)                  # ('circle', 0.97)

# --- REINFORCEMENT: learn from consequences ---
agent = brain.get_agent("maze", actions=["up", "down", "left", "right"])
agent.learn(state, action, reward, next_state)

# --- COGNITION ---
print(brain.think())        # one cognitive cycle
print(brain.sleep())        # memory consolidation + inference
print(brain.introspect())   # full self-report

# --- PERSISTENCE: the brain survives restarts ---
brain.save("./atlas_state")
brain = Brain.load("./atlas_state")
```

## What's inside

| Module | Biological analogue | What it does |
|---|---|---|
| `neurosense.eyes` | Retina + visual cortex | Edge detection (Sobel), Harris corners, blob detection, Gaussian blur, image signatures, one-shot recognition |
| `neurosense.ears` | Cochlea + auditory cortex | FFT spectra, spectrograms, mel filterbanks, pitch detection (autocorrelation), onset detection, note naming, WAV loading (stdlib) |
| `neurosense.neurons` | Neurons + synapses | Backprop networks (Dense/Dropout, ReLU/tanh/sigmoid, SGD/Momentum/Adam, MSE/cross-entropy), Hebbian learning (Oja's rule), Hopfield associative memory, spiking (leaky integrate-and-fire) networks with STDP-like plasticity |
| `neurosense.brain` | Prefrontal cortex + hippocampus | Working memory (decay + displacement), episodic memory (recall, consolidation, forgetting), attention (novelty + habituation), the `Brain` orchestrator |
| `neurosense.knowledge` | Semantic memory | Knowledge graph of (subject, relation, object) facts with confidence, spreading activation, path finding, forward-chaining inference engine with custom rules |
| `neurosense.learning` | Basal ganglia | Tabular Q-learning (epsilon-greedy + decay), KMeans (k-means++), Self-Organizing Maps |
| `neurosense.language` | Language cortex | Tokenizer, TF-IDF similarity, co-occurrence association, n-gram generation, pattern-based fact extraction from English |

## Examples

```bash
python examples/demo_full_agent.py   # everything together
python examples/demo_language.py    # reading, reasoning, generation
python examples/demo_learning.py    # Q-learning solves a maze
python examples/demo_neurons.py     # XOR, Hebbian, Hopfield, spiking
```

## Design principles

1. **No black boxes.** Every algorithm is readable, commented source.
2. **No network calls.** The library never touches the internet.
3. **No emotions.** Cognition only: perceive, remember, reason, learn, decide.
4. **It only knows what it learns.** Empty at birth; grows with experience.

## Run the tests

```bash
python tests/test_all.py
```

## License

MIT — see `LICENSE`.
