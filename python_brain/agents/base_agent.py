"""
base_agent.py
=============
Base class for all Veilcrean Agents.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, Optional
from ..utils.logger import get_logger
from ..skills import GLOBAL_SKILL_ENGINE, GLOBAL_SKILL_REGISTRY
from ..skills.core import SkillResult

class BaseAgent:
    name = "base_agent"
    skill_role = "all"

    def __init__(self):
        self.log = get_logger(self.name)
        self.skill_engine = GLOBAL_SKILL_ENGINE
        self.skill_registry = GLOBAL_SKILL_REGISTRY

    def available_skills(self, executable_only: bool = False):
        """Return skill definitions mapped to this agent/sub-agent role."""
        role = getattr(self, "skill_role", None) or self.name
        return self.skill_registry.loadout_for_agent(role, executable_only=executable_only)

    def use_skill(self, skill_id: str, context: Dict[str, Any]) -> SkillResult:
        """Execute one project skill against the current context."""
        return self.skill_engine.run(skill_id, context)

    def use_skills(self, skill_ids: Iterable[str], context: Dict[str, Any]) -> Dict[str, SkillResult]:
        """Execute multiple project skills against the current context."""
        return self.skill_engine.run_many(skill_ids, context)

    def use_role_skills(self, context: Dict[str, Any], max_skills: Optional[int] = None) -> Dict[str, SkillResult]:
        """Execute this agent's executable role loadout."""
        role = getattr(self, "skill_role", None) or self.name
        return self.skill_engine.run_for_agent(role, context, executable_only=True, max_skills=max_skills)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the context and return insights/decisions."""
        raise NotImplementedError
