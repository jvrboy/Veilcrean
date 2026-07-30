"""Full NeuroSense agent demo."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neurosense import NeuroSenseAgent
from neurosense.ears import tone


def main() -> None:
    agent = NeuroSenseAgent()
    image = np.eye(16)
    audio = tone(440, duration=0.1)
    thought = agent.step({"text": "calm focused learning", "image": image, "audio": audio})
    print(thought.summary)
    print(agent.act(thought))


if __name__ == "__main__":
    main()
