"""KnowledgeGraph — semantic long-term memory.

Facts are (subject, relation, object) triples with confidence scores.
Supports querying, spreading activation, path finding, and JSON persistence.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    obj: str
    confidence: float = 1.0

    def __str__(self) -> str:
        return f"{self.subject} --{self.relation}--> {self.obj} ({self.confidence:.2f})"


class KnowledgeGraph:
    """A growing graph of everything the brain knows.

    >>> kg = KnowledgeGraph()
    >>> kg.add("dog", "is_a", "mammal")
    >>> kg.add("mammal", "has", "fur")
    >>> kg.query(subject="dog")
    [Fact(subject='dog', relation='is_a', obj='mammal', confidence=1.0)]
    """

    def __init__(self):
        self._facts: dict[tuple[str, str, str], Fact] = {}
        self._by_subject: dict[str, set] = defaultdict(set)
        self._by_object: dict[str, set] = defaultdict(set)

    # ------------------------------------------------------------------ #
    def add(self, subject: str, relation: str, obj: str,
            confidence: float = 1.0) -> Fact:
        subject, relation, obj = subject.lower(), relation.lower(), obj.lower()
        key = (subject, relation, obj)
        existing = self._facts.get(key)
        if existing is not None:
            # Repetition strengthens belief (bounded reinforcement)
            confidence = min(1.0, existing.confidence + 0.1 * confidence)
        fact = Fact(subject, relation, obj, confidence)
        self._facts[key] = fact
        self._by_subject[subject].add(key)
        self._by_object[obj].add(key)
        return fact

    def forget(self, subject: str, relation: str, obj: str) -> bool:
        key = (subject.lower(), relation.lower(), obj.lower())
        if key in self._facts:
            del self._facts[key]
            self._by_subject[key[0]].discard(key)
            self._by_object[key[2]].discard(key)
            return True
        return False

    # ------------------------------------------------------------------ #
    def query(self, subject: str | None = None, relation: str | None = None,
              obj: str | None = None) -> list[Fact]:
        """Pattern-match facts. None acts as a wildcard."""
        if subject is not None:
            keys = self._by_subject.get(subject.lower(), set())
        elif obj is not None:
            keys = self._by_object.get(obj.lower(), set())
        else:
            keys = self._facts.keys()
        results = []
        for key in keys:
            fact = self._facts[key]
            if relation is not None and fact.relation != relation.lower():
                continue
            if obj is not None and fact.obj != obj.lower():
                continue
            if subject is not None and fact.subject != subject.lower():
                continue
            results.append(fact)
        return sorted(results, key=lambda f: -f.confidence)

    def entities(self) -> set[str]:
        ents = set()
        for s, _, o in self._facts:
            ents.add(s)
            ents.add(o)
        return ents

    def __len__(self) -> int:
        return len(self._facts)

    def __contains__(self, triple: tuple[str, str, str]) -> bool:
        s, r, o = triple
        return (s.lower(), r.lower(), o.lower()) in self._facts

    # ------------------------------------------------------------------ #
    def find_path(self, start: str, goal: str,
                  max_depth: int = 6) -> list[Fact] | None:
        """Breadth-first chain of facts connecting two concepts."""
        start, goal = start.lower(), goal.lower()
        if start == goal:
            return []
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for key in self._by_subject.get(node, ()):
                fact = self._facts[key]
                if fact.obj == goal:
                    return path + [fact]
                if fact.obj not in seen:
                    seen.add(fact.obj)
                    queue.append((fact.obj, path + [fact]))
        return None

    def spread_activation(self, seed: str, decay: float = 0.5,
                          max_depth: int = 3) -> dict[str, float]:
        """Which concepts light up when thinking about `seed`?

        Returns {concept: activation} — the computational analogue of
        free association.
        """
        seed = seed.lower()
        activation = {seed: 1.0}
        frontier = {seed}
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                energy = activation[node] * decay
                if energy < 0.05:
                    continue
                neighbors = ([self._facts[k].obj
                              for k in self._by_subject.get(node, ())] +
                             [self._facts[k].subject
                              for k in self._by_object.get(node, ())])
                for nb in neighbors:
                    if activation.get(nb, 0.0) < energy:
                        activation[nb] = energy
                        next_frontier.add(nb)
            frontier = next_frontier
        activation.pop(seed, None)
        return dict(sorted(activation.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([asdict(fact) for fact in self._facts.values()], f)

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        kg = cls()
        with open(path) as f:
            for item in json.load(f):
                kg.add(item["subject"], item["relation"], item["obj"],
                       item.get("confidence", 1.0))
        return kg
