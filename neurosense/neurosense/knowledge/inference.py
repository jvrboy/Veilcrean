"""Simple rule-based inference over :mod:`neurosense.knowledge.graph`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .graph import KnowledgeGraph, Triple


@dataclass(frozen=True)
class Rule:
    if_predicate: str
    then_predicate: str


class InferenceEngine:
    """Infer transitive and alias relationships."""

    def __init__(self, graph: KnowledgeGraph | None = None) -> None:
        self.graph = graph or KnowledgeGraph()
        self.rules: list[Rule] = [Rule("is_a", "is_a"), Rule("part_of", "part_of")]

    def add_rule(self, if_predicate: str, then_predicate: str | None = None) -> None:
        self.rules.append(Rule(if_predicate, then_predicate or if_predicate))

    def transitive_closure(self, predicate: str = "is_a") -> list[Triple]:
        inferred: set[Triple] = set()
        changed = True
        while changed:
            changed = False
            triples = set(self.graph.query(predicate=predicate)) | {t for t in inferred if t.predicate == predicate}
            for a in triples:
                for b in triples:
                    if a.object == b.subject:
                        t = Triple(a.subject, predicate, b.object)
                        if t not in triples and t not in inferred:
                            inferred.add(t)
                            changed = True
        return sorted(inferred)

    def infer(self) -> list[Triple]:
        inferred: set[Triple] = set()
        for rule in self.rules:
            for t in self.transitive_closure(rule.if_predicate):
                inferred.add(Triple(t.subject, rule.then_predicate, t.object))
        for t in sorted(inferred):
            self.graph.add_edge(t.subject, t.predicate, t.object)
        return sorted(inferred)

    def query(self, *args, **kwargs) -> list[Triple]:
        self.infer()
        return self.graph.query(*args, **kwargs)

    def explain(self, subject: str, object: str, predicate: str = "is_a") -> str:
        path = self.graph.shortest_path(subject, object)
        if path:
            return " -> ".join(path)
        for triple in self.transitive_closure(predicate):
            if triple.subject == subject and triple.object == object:
                return f"{subject} {predicate} ... {object}"
        return "no inference path found"
