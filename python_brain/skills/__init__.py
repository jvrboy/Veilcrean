"""Agent skill system.

Agents can inspect the complete trader skill catalog and execute the practical
skills directly from context:

    from python_brain.skills import GLOBAL_SKILL_ENGINE
    result = GLOBAL_SKILL_ENGINE.run("ta_trend_indicators", {"buffers": buffers})
"""
from __future__ import annotations

from .catalog import ALL_SKILL_DEFINITIONS, INSTITUTIONAL_TRADER_SKILLS, TECHNICAL_TRADER_SKILLS
from .core import BaseSkill, SkillDefinition, SkillEngine, SkillRegistry, SkillResult
from .executable import EXECUTABLE_SKILLS


def build_default_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register_many(ALL_SKILL_DEFINITIONS)
    for skill in EXECUTABLE_SKILLS:
        registry.register_skill(skill)
    return registry


GLOBAL_SKILL_REGISTRY = build_default_registry()
GLOBAL_SKILL_ENGINE = SkillEngine(GLOBAL_SKILL_REGISTRY)

__all__ = [
    "SkillDefinition",
    "SkillResult",
    "BaseSkill",
    "SkillRegistry",
    "SkillEngine",
    "TECHNICAL_TRADER_SKILLS",
    "INSTITUTIONAL_TRADER_SKILLS",
    "ALL_SKILL_DEFINITIONS",
    "EXECUTABLE_SKILLS",
    "build_default_registry",
    "GLOBAL_SKILL_REGISTRY",
    "GLOBAL_SKILL_ENGINE",
]
