"""
core.py
=======
Runtime skill system for Veilcrean agents and sub-agents.

A *skill* is deliberately broader than an analysis tool: it can be a knowledge
capability, a process checklist, or an executable market-analysis routine. The
registry exposes the whole professional trader curriculum to agents while the
engine runs the executable subset against live context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class SkillDefinition:
    """Metadata describing a skill agents can discover and request."""

    id: str
    name: str
    category: str
    description: str = ""
    components: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    agent_roles: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    executable: bool = False

    def matches(self, query: str) -> bool:
        q = query.lower().strip()
        haystack = " ".join(
            [self.id, self.name, self.category, self.description, *self.components, *self.tags, *self.agent_roles]
        ).lower()
        return q in haystack


@dataclass
class SkillResult:
    """Output from an executable skill."""

    skill_id: str
    name: str
    category: str
    score: float = 0.0
    confidence: float = 0.0
    direction: str = "NEUTRAL"
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    signals: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.score = float(np.clip(np.nan_to_num(self.score), -1.0, 1.0))
        self.confidence = float(np.clip(np.nan_to_num(self.confidence), 0.0, 1.0))
        if not self.direction or self.direction == "NEUTRAL":
            if self.score > 0.15:
                self.direction = "BULLISH"
            elif self.score < -0.15:
                self.direction = "BEARISH"
            else:
                self.direction = "NEUTRAL"

    def is_valid(self) -> bool:
        return not self.errors

    def is_actionable(self, min_confidence: float = 0.55, min_abs_score: float = 0.25) -> bool:
        return self.is_valid() and self.confidence >= min_confidence and abs(self.score) >= min_abs_score

    def as_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "category": self.category,
            "score": self.score,
            "confidence": self.confidence,
            "direction": self.direction,
            "features": dict(self.features),
            "metadata": dict(self.metadata),
            "signals": list(self.signals),
            "errors": list(self.errors),
        }


class ExecutableSkill(Protocol):
    definition: SkillDefinition

    def run(self, context: Dict[str, Any]) -> SkillResult: ...


class BaseSkill:
    """Base class for executable skills."""

    definition: SkillDefinition

    @property
    def id(self) -> str:
        return self.definition.id

    def unavailable(self, reason: str, **metadata: Any) -> SkillResult:
        return SkillResult(
            skill_id=self.definition.id,
            name=self.definition.name,
            category=self.definition.category,
            score=0.0,
            confidence=0.0,
            metadata={"available": False, "reason": reason, **metadata},
            errors=[reason],
        )

    def result(
        self,
        score: float = 0.0,
        confidence: float = 0.5,
        *,
        direction: str = "NEUTRAL",
        features: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        signals: Optional[Sequence[str]] = None,
    ) -> SkillResult:
        return SkillResult(
            skill_id=self.definition.id,
            name=self.definition.name,
            category=self.definition.category,
            score=score,
            confidence=confidence,
            direction=direction,
            features=features or {},
            metadata=metadata or {},
            signals=list(signals or []),
        )

    def run(self, context: Dict[str, Any]) -> SkillResult:  # pragma: no cover - abstract
        raise NotImplementedError


class SkillRegistry:
    """Discoverable registry of all skills plus executable implementations."""

    def __init__(self) -> None:
        self._definitions: Dict[str, SkillDefinition] = {}
        self._executors: Dict[str, ExecutableSkill] = {}

    def register_definition(self, definition: SkillDefinition) -> SkillDefinition:
        self._definitions[definition.id] = definition
        return definition

    def register_many(self, definitions: Iterable[SkillDefinition]) -> None:
        for definition in definitions:
            self.register_definition(definition)

    def register_skill(self, skill: ExecutableSkill) -> ExecutableSkill:
        definition = SkillDefinition(
            **{**skill.definition.__dict__, "executable": True}
        )
        self._definitions[definition.id] = definition
        self._executors[definition.id] = skill
        return skill

    def get(self, skill_id: str) -> SkillDefinition:
        return self._definitions[skill_id]

    def executor(self, skill_id: str) -> Optional[ExecutableSkill]:
        return self._executors.get(skill_id)

    def definitions(self) -> List[SkillDefinition]:
        return sorted(self._definitions.values(), key=lambda d: (d.category, d.id))

    def executable_ids(self) -> List[str]:
        return sorted(self._executors)

    def categories(self) -> List[str]:
        return sorted({d.category for d in self._definitions.values()})

    def search(
        self,
        query: str | None = None,
        *,
        category: str | None = None,
        tags: Iterable[str] | None = None,
        agent_role: str | None = None,
        executable: bool | None = None,
    ) -> List[SkillDefinition]:
        tag_set = {t.lower() for t in (tags or [])}
        role = agent_role.lower() if agent_role else None
        out: List[SkillDefinition] = []
        for definition in self._definitions.values():
            if query and not definition.matches(query):
                continue
            if category and definition.category != category:
                continue
            if executable is not None and definition.executable != executable:
                continue
            if tag_set and not tag_set.intersection({t.lower() for t in definition.tags}):
                continue
            if role and role not in {r.lower() for r in definition.agent_roles} and "all" not in {
                r.lower() for r in definition.agent_roles
            }:
                continue
            out.append(definition)
        return sorted(out, key=lambda d: (d.category, d.id))

    def loadout_for_agent(self, agent_role: str, executable_only: bool = False) -> List[SkillDefinition]:
        return self.search(agent_role=agent_role, executable=True if executable_only else None)

    def summary(self) -> Dict[str, Any]:
        by_category: Dict[str, int] = {}
        for definition in self._definitions.values():
            by_category[definition.category] = by_category.get(definition.category, 0) + 1
        return {
            "total_skills": len(self._definitions),
            "executable_skills": len(self._executors),
            "categories": by_category,
        }


class SkillEngine:
    """Runs skills and builds agent-facing reports."""

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def run(self, skill_id: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        context = dict(context or {})
        skill = self.registry.executor(skill_id)
        definition = self.registry.get(skill_id)
        if skill is None:
            return SkillResult(
                skill_id=definition.id,
                name=definition.name,
                category=definition.category,
                score=0.0,
                confidence=0.0,
                metadata={"available": False, "reason": "skill is knowledge-only"},
                errors=["skill is knowledge-only"],
            )
        try:
            return skill.run(context)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return SkillResult(
                skill_id=definition.id,
                name=definition.name,
                category=definition.category,
                score=0.0,
                confidence=0.0,
                metadata={"available": False, "exception": str(exc)},
                errors=[f"exception: {exc}"],
            )

    def run_many(self, skill_ids: Iterable[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, SkillResult]:
        return {skill_id: self.run(skill_id, context) for skill_id in skill_ids}

    def run_for_agent(
        self,
        agent_role: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        executable_only: bool = True,
        max_skills: Optional[int] = None,
    ) -> Dict[str, SkillResult]:
        loadout = self.registry.loadout_for_agent(agent_role, executable_only=executable_only)
        if max_skills is not None:
            loadout = loadout[:max_skills]
        return self.run_many([d.id for d in loadout if d.executable], context)

    def aggregate(self, results: Dict[str, SkillResult]) -> Dict[str, Any]:
        category_scores: Dict[str, List[float]] = {}
        category_conf: Dict[str, List[float]] = {}
        for result in results.values():
            if not result.is_valid():
                continue
            category_scores.setdefault(result.category, []).append(result.score * max(result.confidence, 0.05))
            category_conf.setdefault(result.category, []).append(max(result.confidence, 0.05))

        categories: Dict[str, Dict[str, float]] = {}
        weighted_sum = 0.0
        total_weight = 0.0
        for category, scores in category_scores.items():
            weights = category_conf[category]
            score = float(sum(scores) / max(sum(weights), 1e-9))
            confidence = float(np.mean(weights))
            categories[category] = {"score": score, "confidence": confidence}
            weighted_sum += score * confidence
            total_weight += confidence

        actionable = [r.as_dict() for r in results.values() if r.is_actionable()]
        return {
            "aggregate_score": float(np.clip(weighted_sum / max(total_weight, 1e-9), -1.0, 1.0)),
            "categories": categories,
            "actionable": actionable,
            "valid_count": sum(1 for r in results.values() if r.is_valid()),
            "error_count": sum(1 for r in results.values() if not r.is_valid()),
        }

    def catalog_summary(self) -> Dict[str, Any]:
        return self.registry.summary()
