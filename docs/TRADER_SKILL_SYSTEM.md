# Veilcrean Trader Skill System

This project now includes a project-wide skill layer that agents and sub-agents can discover and execute.

## What was added

- `python_brain.skills` package
  - `SkillDefinition` — metadata for every skill in the technical + institutional trader maps.
  - `SkillRegistry` — searchable registry by category, tag, role, or executable status.
  - `SkillEngine` — runs executable skills against the agent context.
  - `GLOBAL_SKILL_REGISTRY` / `GLOBAL_SKILL_ENGINE` — shared by all agents.
- 104 cataloged skills:
  - 50 technical-analysis trader skills.
  - 54 professional institutional trader skills.
- 26 executable skills, including:
  - multi-timeframe analysis
  - candlestick recognition
  - reversal/continuation patterns
  - trend, momentum, volatility, and volume indicators
  - support/resistance
  - price action and market structure
  - Smart Money Concepts
  - divergence
  - position sizing, stops, R:R, trade planning, entry/management/exit
  - institutional volume profile
  - order-flow proxy analysis
  - sentiment and execution-quality checks
  - backtesting metrics and gap analysis
- New analysis tools powered by the skill engine:
  - `TraderSkillStackTool`
  - `AdvancedPatternSkillTool`
  - `InstitutionalFlowSkillTool`
  - `RiskPlanningSkillTool`
- `BaseAgent` now exposes helper methods:
  - `available_skills(executable_only=False)`
  - `use_skill(skill_id, context)`
  - `use_skills(skill_ids, context)`
  - `use_role_skills(context, max_skills=None)`
- `SkillAgent` sub-agent is wired into `CoordinatorAgent` and publishes `skill_report`.

## Example usage

```python
from python_brain.skills import GLOBAL_SKILL_ENGINE

result = GLOBAL_SKILL_ENGINE.run("ta_trend_indicators", {"buffers": buffers})
print(result.score, result.direction, result.signals)
```

Inside any agent:

```python
class MyAgent(BaseAgent):
    skill_role = "analyst"

    def run(self, context):
        trend = self.use_skill("ta_trend_indicators", context)
        smc = self.use_skill("ta_smart_money_concepts", context)
        return {"trend": trend.as_dict(), "smc": smc.as_dict()}
```

## Agent role loadouts

Skills are mapped to roles such as:

- `analyst`
- `strategist`
- `risk_officer`
- `execution`
- `sentiment`
- `research`
- `portfolio`
- `monitoring`
- `optimization`
- `broker`
- `coordinator`
- `all`

Agents can run their executable role skills with:

```python
results = self.use_role_skills(context)
```

## Notes

The skill layer is intentionally lightweight and deterministic. It uses the repository's existing OHLCV buffers and context dictionaries. It does not require paid institutional feeds; where order-book or intermarket data is missing, skills return low-confidence metadata instead of failing.
