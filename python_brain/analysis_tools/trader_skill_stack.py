"""Analysis tools backed by the project-wide trader skill engine."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult
from ..skills import GLOBAL_SKILL_ENGINE
from ..skills.core import SkillResult


def _context(buffers: Dict[str, pd.DataFrame], ctx: dict) -> dict:
    out = dict(ctx)
    out["buffers"] = buffers
    return out


def _to_tool_result(tool_name: str, skill_results: Dict[str, SkillResult], metadata_extra: dict | None = None) -> ToolResult:
    aggregate = GLOBAL_SKILL_ENGINE.aggregate(skill_results)
    features: dict[str, float] = {
        "skill_valid_count": float(aggregate["valid_count"]),
        "skill_error_count": float(aggregate["error_count"]),
        "skill_actionable_count": float(len(aggregate["actionable"])),
    }
    for skill_id, result in skill_results.items():
        safe_id = skill_id.replace("-", "_")
        features[f"{safe_id}__score"] = result.score
        features[f"{safe_id}__confidence"] = result.confidence
        features[f"{safe_id}__actionable"] = float(result.is_actionable())
        for key, value in (result.features or {}).items():
            try:
                features[f"{safe_id}__{key}"] = float(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0))
            except Exception:
                features[f"{safe_id}__{key}"] = 0.0
    metadata = {
        "skills": {skill_id: result.as_dict() for skill_id, result in skill_results.items()},
        "skill_categories": aggregate["categories"],
        "actionable": aggregate["actionable"],
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    errors = [err for result in skill_results.values() for err in result.errors]
    return ToolResult(
        tool_name=tool_name,
        score=aggregate["aggregate_score"],
        confidence=float(np.mean([r.confidence for r in skill_results.values()])) if skill_results else 0.0,
        features=features,
        metadata=metadata,
        errors=[] if any(r.is_valid() for r in skill_results.values()) else errors,
    )


class TraderSkillStackTool(BaseTool):
    """Runs the full executable professional trader skill stack."""

    name = "trader_skill_stack"
    skill_ids = [
        "ta_timeframe_analysis",
        "ta_candlestick_patterns",
        "ta_reversal_patterns",
        "ta_continuation_patterns",
        "ta_trend_indicators",
        "ta_momentum_indicators",
        "ta_volatility_indicators",
        "ta_volume_indicators",
        "ta_support_resistance",
        "ta_price_action",
        "ta_smart_money_concepts",
        "ta_divergence_analysis",
        "ta_intermarket_analysis",
        "ta_position_sizing",
        "ta_stop_loss",
        "ta_risk_reward",
        "ta_trading_plan",
        "ta_trade_entry",
        "ta_trade_management",
        "ta_trade_exit",
        "ta_recordkeeping_metrics",
        "ta_gap_analysis",
        "inst_volume_profile",
        "inst_order_flow",
        "inst_sentiment_analysis",
        "inst_execution_quality",
    ]

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        context = _context(buffers, ctx)
        results = GLOBAL_SKILL_ENGINE.run_many(self.skill_ids, context)
        return _to_tool_result("TraderSkillStackTool", results, {"tool_purpose": "complete executable trader skill stack"})


class AdvancedPatternSkillTool(BaseTool):
    """Runs advanced pattern, price action, SMC, divergence, and gap skills."""

    name = "advanced_pattern_skills"
    skill_ids = [
        "ta_candlestick_patterns",
        "ta_reversal_patterns",
        "ta_continuation_patterns",
        "ta_price_action",
        "ta_smart_money_concepts",
        "ta_divergence_analysis",
        "ta_gap_analysis",
    ]

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        results = GLOBAL_SKILL_ENGINE.run_many(self.skill_ids, _context(buffers, ctx))
        return _to_tool_result("AdvancedPatternSkillTool", results, {"tool_purpose": "advanced chart pattern and SMC skill layer"})


class InstitutionalFlowSkillTool(BaseTool):
    """Runs institutional volume profile, order flow, VWAP/volume, and execution-quality skills."""

    name = "institutional_flow_skills"
    skill_ids = [
        "ta_volume_indicators",
        "inst_volume_profile",
        "inst_order_flow",
        "inst_execution_quality",
        "ta_intermarket_analysis",
    ]

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        results = GLOBAL_SKILL_ENGINE.run_many(self.skill_ids, _context(buffers, ctx))
        return _to_tool_result("InstitutionalFlowSkillTool", results, {"tool_purpose": "institutional flow and execution skill layer"})


class RiskPlanningSkillTool(BaseTool):
    """Runs sizing, stop, R:R, plan, entry, management, and exit skills."""

    name = "risk_planning_skills"
    skill_ids = [
        "ta_position_sizing",
        "ta_stop_loss",
        "ta_risk_reward",
        "ta_trading_plan",
        "ta_trade_entry",
        "ta_trade_management",
        "ta_trade_exit",
    ]

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        results = GLOBAL_SKILL_ENGINE.run_many(self.skill_ids, _context(buffers, ctx))
        return _to_tool_result("RiskPlanningSkillTool", results, {"tool_purpose": "risk planning and execution-management skill layer"})
