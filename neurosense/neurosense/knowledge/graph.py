"""Directed labeled graph for symbolic knowledge."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, order=True)
class Triple:
    subject: str
    predicate: str
    object: str


class KnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: set[str] = set()
        self.edges: set[Triple] = set()
        self.attributes: dict[str, dict[str, Any]] = defaultdict(dict)

    def add_node(self, node: str, **attrs: Any) -> str:
        node = str(node)
        self.nodes.add(node)
        self.attributes[node].update(attrs)
        return node

    def add_edge(self, subject: str, predicate: str, object: str) -> Triple:
        triple = Triple(str(subject), str(predicate), str(object))
        self.nodes.update([triple.subject, triple.object])
        self.edges.add(triple)
        return triple

    add_triple = add_edge

    def neighbors(self, node: str, predicate: str | None = None) -> list[str]:
        return sorted(t.object for t in self.edges if t.subject == node and (predicate is None or t.predicate == predicate))

    def incoming(self, node: str, predicate: str | None = None) -> list[str]:
        return sorted(t.subject for t in self.edges if t.object == node and (predicate is None or t.predicate == predicate))

    def query(self, subject: str | None = None, predicate: str | None = None, object: str | None = None) -> list[Triple]:
        return sorted(
            t
            for t in self.edges
            if (subject is None or t.subject == subject)
            and (predicate is None or t.predicate == predicate)
            and (object is None or t.object == object)
        )

    def triples(self) -> list[Triple]:
        return sorted(self.edges)

    def shortest_path(self, source: str, target: str) -> list[str]:
        if source == target:
            return [source]
        q = deque([(source, [source])])
        seen = {source}
        while q:
            node, path = q.popleft()
            for nxt in self.neighbors(node):
                if nxt in seen:
                    continue
                if nxt == target:
                    return path + [nxt]
                seen.add(nxt)
                q.append((nxt, path + [nxt]))
        return []

    def __len__(self) -> int:
        return len(self.edges)
