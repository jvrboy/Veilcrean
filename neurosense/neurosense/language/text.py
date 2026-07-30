"""Lightweight text processing utilities."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
_POSITIVE = {"good", "great", "calm", "focused", "happy", "learn", "learning", "safe", "win", "gain", "clear"}
_NEGATIVE = {"bad", "angry", "sad", "loss", "lose", "danger", "risk", "fear", "noisy", "error"}


def tokenize(text: str) -> list[str]:
    """Lowercase regex tokenization."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(str(text))]


def bag_of_words(text: str, vocabulary: Iterable[str] | None = None) -> dict[str, int]:
    counts = Counter(tokenize(text))
    if vocabulary is None:
        return dict(counts)
    return {token: counts.get(token, 0) for token in vocabulary}


def sentiment(text: str) -> float:
    """Return a simple lexical sentiment score in ``[-1, 1]``."""
    toks = tokenize(text)
    if not toks:
        return 0.0
    score = sum(1 for t in toks if t in _POSITIVE) - sum(1 for t in toks if t in _NEGATIVE)
    return float(max(-1.0, min(1.0, score / math.sqrt(len(toks)))))


def summarize(text: str, max_words: int = 16) -> str:
    toks = tokenize(text)
    if not toks:
        return ""
    if len(toks) <= max_words:
        return " ".join(toks)
    return " ".join(toks[:max_words]) + " ..."


@dataclass
class TextVectorizer:
    vocabulary: list[str] = field(default_factory=list)

    def fit(self, texts: Iterable[str]) -> "TextVectorizer":
        vocab = sorted({token for text in texts for token in tokenize(text)})
        self.vocabulary = vocab
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        if not self.vocabulary:
            raise RuntimeError("fit must be called before transform")
        rows = []
        for text in texts:
            bow = bag_of_words(text, self.vocabulary)
            rows.append([bow[token] for token in self.vocabulary])
        return np.asarray(rows, dtype=float)

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        return self.fit(texts).transform(texts)


@dataclass
class TextSensor:
    max_words: int = 16

    def perceive(self, text: str) -> dict:
        tokens = tokenize(text)
        score = sentiment(text)
        return {
            "modality": "language",
            "summary": summarize(text, self.max_words),
            "features": {
                "token_count": len(tokens),
                "unique_tokens": len(set(tokens)),
                "sentiment": score,
                "avg_token_length": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
            },
        }

    process = perceive
