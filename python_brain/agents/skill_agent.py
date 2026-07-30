"""
skill_agent.py
==============
Sub-agent that exposes and executes the project-wide trader skill system.
"""
from __future__ import annotations

from typing import Any, Dict

from .base_agent import BaseAgent


class SkillAgent(BaseAgent):
    """Runs executable skills and publishes a structured skill report."""

    skill_role = "coordinator"
    name = "skill_agent"

    DEFAULT_ROLES = (
        "analyst",
        "strategist",
        "risk_officer",
        "execution",
        "sentiment",
        "research",
        "portfolio",
        "monitoring",
        "optimization",
        "broker",
        "coordinator",
    )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        roles = tuple(context.get("skill_roles") or self.DEFAULT_ROLES)
        requested = context.get("skill_ids")

        if requested:
            results = self.skill_engine.run_many(requested, context)
        else:
            skill_ids: list[str] = []
            for role in roles:
                for definition in self.skill_registry.loadout_for_agent(role, executable_only=True):
                    if definition.id not in skill_ids:
                        skill_ids.append(definition.id)
            results = self.skill_engine.run_many(skill_ids, context)

        aggregate = self.skill_engine.aggregate(results)
        inventory = {
            role: [d.id for d in self.skill_registry.loadout_for_agent(role, executable_only=False)]
            for role in roles
        }
        executable_inventory = {
            role: [d.id for d in self.skill_registry.loadout_for_agent(role, executable_only=True)]
            for role in roles
        }

        return {
            "skill_report": {
                "catalog": self.skill_engine.catalog_summary(),
                "aggregate": aggregate,
                "results": {skill_id: result.as_dict() for skill_id, result in results.items()},
                "inventory": inventory,
                "executable_inventory": executable_inventory,
            },
            "skill_aggregate_score": aggregate["aggregate_score"],
            "skill_actionable": aggregate["actionable"],
        }
