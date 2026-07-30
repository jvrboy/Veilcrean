"""Text processing demo."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurosense.language import TextSensor, TextVectorizer, sentiment, tokenize


def main() -> None:
    text = "Calm focused learning creates clear and safe decisions."
    print(tokenize(text))
    print("sentiment", sentiment(text))
    print(TextSensor().perceive(text))
    print(TextVectorizer().fit_transform([text, "noisy error risk"]))


if __name__ == "__main__":
    main()
