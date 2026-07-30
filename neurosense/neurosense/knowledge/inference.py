"""InferenceEngine — reasoning over the knowledge graph.

Forward-chaining rule engine with built-in transitive inheritance:
if dog is_a mammal and mammal has fur, the brain concludes dog has fur.
Users can add custom rules of the form:
    (X, r1, Y) and (Y, r2, Z)  =>  (X, r3, Z)
"""

from __future__ import annotations

from dataclasses import dataclass

from .graph import KnowledgeGraph, Fact


@dataclass(frozen=True)
class Rule:
    """A chaining rule: premise1_rel + premise2_rel => conclusion_rel.

    Semantics: if (X premise1 Y) and (Y premise2 Z) then (X conclusion Z).
    """

    premise1: str
    premise2: str
    conclusion: str
    confidence_factor: float = 0.9

    def __str__(self) -> str:
        return (f"(X {self.premise1} Y) & (Y {self.premise2} Z) "
                f"=> (X {self.conclusion} Z)")


DEFAULT_RULES = [
    # Category inheritance: dog is_a mammal, mammal has fur => dog has fur
    Rule("is_a", "has", "has"),
    Rule("is_a", "can", "can"),
    Rule("is_a", "is_a", "is_a"),
    Rule("part_of", "part_of", "part_of"),
    Rule("located_in", "located_in", "located_in"),
]


class InferenceEngine:
    """Derives new knowledge from existing knowledge.

    >>> engine = InferenceEngine(kg)
    >>> new_facts = engine.infer()          # forward chain until fixpoint
    >>> engine.ask("dog", "has", "fur")     # (True, confidence, chain)
    """

    def __init__(self, graph: KnowledgeGraph,
                 rules: list[Rule] | None = None):
        self.graph = graph
        self.rules = list(rules) if rules is not None else list(DEFAULT_RULES)

    def add_rule(self, premise1: str, premise2: str, conclusion: str,
                 confidence_factor: float = 0.9) -> Rule:
        rule = Rule(premise1, premise2, conclusion, confidence_factor)
        self.rules.append(rule)
        return rule

    # ------------------------------------------------------------------ #
    def infer(self, max_iterations: int = 10) -> list[Fact]:
        """Forward-chain all rules until no new facts appear (fixpoint)."""
        derived: list[Fact] = []
        for _ in range(max_iterations):
            new_this_round = []
            for rule in self.rules:
                for f1 in self.graph.query(relation=rule.premise1):
                    for f2 in self.graph.query(subject=f1.obj,
                                               relation=rule.premise2):
                        triple = (f1.subject, rule.conclusion, f2.obj)
                        if f1.subject == f2.obj:
                            continue
                        if triple in self.graph:
                            continue
                        conf = (f1.confidence * f2.confidence
                                * rule.confidence_factor)
                        if conf < 0.1:
                            continue
                        new_this_round.append(
                            self.graph.add(*triple, confidence=conf))
            if not new_this_round:
                break
            derived.extend(new_this_round)
        return derived

    # ------------------------------------------------------------------ #
    def ask(self, subject: str, relation: str,
            obj: str) -> tuple[bool, float, list[Fact]]:
        """Is (subject, relation, obj) true? Returns (answer, confidence, proof).

        Checks stored facts first, then runs inference, then searches for
        an explanatory chain between the two concepts.
        """
        direct = self.graph.query(subject=subject, relation=relation, obj=obj)
        if direct:
            return True, direct[0].confidence, direct
        self.infer()
        direct = self.graph.query(subject=subject, relation=relation, obj=obj)
        if direct:
            chain = self.graph.find_path(subject, obj) or direct
            return True, direct[0].confidence, chain
        return False, 0.0, []

    def why(self, subject: str, obj: str) -> str:
        """Human-readable explanation of how two concepts connect."""
        path = self.graph.find_path(subject, obj)
        if path is None:
            return f"I see no connection between '{subject}' and '{obj}'."
        if not path:
            return f"'{subject}' and '{obj}' are the same concept."
        steps = " ; ".join(f"{f.subject} {f.relation} {f.obj}" for f in path)
        return f"Because: {steps}."
