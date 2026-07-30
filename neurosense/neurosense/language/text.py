"""LanguageCortex — statistical language understanding and generation.

Everything here is learned from the text YOU feed it:
- tokenization and sentence splitting
- TF-IDF document vectors and similarity search
- n-gram language model for original text generation
- co-occurrence based word association
- simple pattern-based fact extraction ("X is a Y", "X can Y", "X has Y")

No pretrained models, no API calls. It knows only what it reads.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "and",
    "or", "but", "not", "no", "it", "its", "this", "that", "these",
    "those", "as", "if", "then", "than", "so", "very", "can", "will",
    "just", "into", "over", "under", "about", "also", "such",
}

_FACT_PATTERNS = [
    (re.compile(r"(\w[\w' -]*?)\s+is\s+an?\s+([\w' -]+)"), "is_a"),
    (re.compile(r"(\w[\w' -]*?)\s+are\s+([\w' -]+)"), "is_a"),
    (re.compile(r"(\w[\w' -]*?)\s+can\s+([\w' -]+)"), "can"),
    (re.compile(r"(\w[\w' -]*?)\s+has\s+([\w' -]+)"), "has"),
    (re.compile(r"(\w[\w' -]*?)\s+have\s+([\w' -]+)"), "has"),
    (re.compile(r"(\w[\w' -]*?)\s+lives?\s+in\s+([\w' -]+)"), "located_in"),
]


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def sentence_split(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


class LanguageCortex:
    """Reads text, builds understanding, answers similarity queries,
    associates words, extracts facts, and generates original sentences.

    >>> lc = LanguageCortex()
    >>> lc.read("The sun is a star. Stars can shine. The sun has heat.")
    >>> lc.extract_facts("The sun is a star.")
    [('sun', 'is_a', 'star')]
    >>> lc.generate(seed="the sun")
    """

    def __init__(self, ngram: int = 3, seed: int | None = None):
        self.n = max(2, ngram)
        self._rng = random.Random(seed)
        self.documents: list[list[str]] = []
        self.doc_texts: list[str] = []
        self.doc_freq: Counter = Counter()
        self.cooccurrence: dict[str, Counter] = defaultdict(Counter)
        self.ngrams: dict[tuple, Counter] = defaultdict(Counter)
        self.vocabulary: Counter = Counter()

    # ------------------------------- reading -------------------------- #
    def read(self, text: str) -> int:
        """Absorb text into all statistical models. Returns tokens read."""
        total = 0
        for sentence in sentence_split(text):
            tokens = tokenize(sentence)
            if not tokens:
                continue
            total += len(tokens)
            self.documents.append(tokens)
            self.doc_texts.append(sentence)
            self.vocabulary.update(tokens)
            self.doc_freq.update(set(tokens))
            # co-occurrence within a +-4 token window
            content = [t for t in tokens if t not in STOPWORDS]
            for i, w in enumerate(content):
                for j in range(max(0, i - 4), min(len(content), i + 5)):
                    if i != j:
                        self.cooccurrence[w][content[j]] += 1
            # n-gram model with sentence boundary markers
            padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i + self.n - 1])
                self.ngrams[context][padded[i + self.n - 1]] += 1
        return total

    # ------------------------------ vectors --------------------------- #
    def vector(self, text: str) -> dict[str, float]:
        """Sparse TF-IDF vector of a text."""
        tokens = tokenize(text)
        tf = Counter(tokens)
        n_docs = max(len(self.documents), 1)
        vec = {}
        for word, count in tf.items():
            idf = math.log((1 + n_docs) / (1 + self.doc_freq.get(word, 0))) + 1
            vec[word] = (count / len(tokens)) * idf
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        num = sum(a[w] * b[w] for w in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0.0

    def most_similar(self, query: str, top: int = 3) -> list[tuple[str, float]]:
        """Find sentences it has read that are most similar to the query."""
        qv = self.vector(query)
        scored = [(text, self._cosine(qv, self.vector(text)))
                  for text in self.doc_texts]
        scored.sort(key=lambda kv: -kv[1])
        return scored[:top]

    def associate(self, word: str, top: int = 5) -> list[tuple[str, int]]:
        """Words most associated with a word (learned co-occurrence)."""
        return self.cooccurrence.get(word.lower(), Counter()).most_common(top)

    # ---------------------------- extraction -------------------------- #
    def extract_facts(self, text: str) -> list[tuple[str, str, str]]:
        """Pull (subject, relation, object) triples from natural language."""
        facts = []
        for sentence in sentence_split(text):
            lowered = sentence.lower().rstrip(".!?")
            for pattern, relation in _FACT_PATTERNS:
                for match in pattern.finditer(lowered):
                    subject = _clean_entity(match.group(1))
                    obj = _clean_entity(match.group(2))
                    if subject and obj and subject != obj:
                        facts.append((subject, relation, obj))
        return facts

    # ---------------------------- generation -------------------------- #
    def generate(self, seed: str = "", max_words: int = 30) -> str:
        """Generate an original sentence from the learned n-gram model."""
        if not self.ngrams:
            return ""
        seed_tokens = tokenize(seed)
        context = tuple((["<s>"] * (self.n - 1) + seed_tokens)[-(self.n - 1):])
        words = list(seed_tokens)
        for _ in range(max_words):
            options = self.ngrams.get(context)
            if not options:
                # back off: relax context to anything that continues
                candidates = [c for c in self.ngrams if c[1:] == context[1:]]
                if not candidates:
                    break
                options = self.ngrams[self._rng.choice(candidates)]
            choices, weights = zip(*options.items())
            word = self._rng.choices(choices, weights=weights)[0]
            if word == "</s>":
                break
            words.append(word)
            context = tuple((list(context) + [word])[1:])
        return (" ".join(words).capitalize() + ".") if words else ""

    def keywords(self, text: str, top: int = 5) -> list[str]:
        vec = self.vector(text)
        ranked = sorted(vec.items(), key=lambda kv: -kv[1])
        return [w for w, _ in ranked if w not in STOPWORDS][:top]


def _singularize(word: str) -> str:
    """Crude but effective English singularization for concept unification.

    Ensures 'mammals have fur' and 'a cat is a mammal' refer to the same
    node in the knowledge graph.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if (len(word) > 3 and word.endswith("s")
            and not word.endswith(("ss", "us", "is"))):
        return word[:-1]
    return word


def _clean_entity(raw: str) -> str:
    tokens = [_singularize(t) for t in tokenize(raw) if t not in STOPWORDS]
    return " ".join(tokens[:3])
