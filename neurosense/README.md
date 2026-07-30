# NeuroSense

NeuroSense is a small, dependency-light Python toolkit for building toy
neuro-symbolic agents. It provides:

- **Eyes**: image normalization, visual feature extraction, and simple motion cues.
- **Ears**: audio normalization, spectral summaries, and synthetic tone helpers.
- **Neurons**: activation functions, dense layers, optimizers, sequential networks,
  spiking neurons, and Hebbian learning utilities.
- **Brain**: attention, memory, and an agent wrapper that combines sensory signals.
- **Knowledge**: a directed knowledge graph and a simple inference engine.
- **Learning**: PCA, k-means, Q-learning, and replay-buffer primitives.
- **Language**: tokenization, bag-of-words vectors, simple sentiment, and summarization.

The package is intentionally compact and deterministic so it can be used in tests,
examples, simulations, and agent prototypes without requiring GPU frameworks.

## Install

```bash
cd neurosense
python -m pip install -e .
```

## Run tests

```bash
cd neurosense
python -m pytest
```

## Quick start

```python
import numpy as np
from neurosense import NeuroSenseAgent

agent = NeuroSenseAgent()
thought = agent.step({
    "text": "calm focused learning",
    "image": np.zeros((16, 16)),
})
print(thought.summary)
```
