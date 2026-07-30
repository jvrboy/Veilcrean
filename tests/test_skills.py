from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def _make_df(n: int = 160, slope: float = 0.001, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(slope, 0.0008, n)
    prices = 1.0 + np.cumsum(rets)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices + rng.normal(0, 0.00015, n),
            "high": prices + np.abs(rng.normal(0.0005, 0.00012, n)),
            "low": prices - np.abs(rng.normal(0.0005, 0.00012, n)),
            "close": prices,
            "volume": rng.integers(100, 1200, n).astype(float),
        },
        index=idx,
    )


def test_skill_catalog_complete_and_searchable():
    from python_brain.skills import GLOBAL_SKILL_REGISTRY

    summary = GLOBAL_SKILL_REGISTRY.summary()
    assert summary["total_skills"] >= 104
    assert summary["executable_skills"] >= 20
    assert GLOBAL_SKILL_REGISTRY.search("smart money")
    assert GLOBAL_SKILL_REGISTRY.loadout_for_agent("analyst", executable_only=True)


def test_skill_engine_runs_executable_skills():
    from python_brain.skills import GLOBAL_SKILL_ENGINE

    buffers = {tf: _make_df(seed=i) for i, tf in enumerate(["M5", "M15", "H1", "H4", "D1"])}
    context = {"buffers": buffers, "price": float(buffers["M15"]["close"].iloc[-1]), "spread": 1.2}
    results = GLOBAL_SKILL_ENGINE.run_many(
        ["ta_timeframe_analysis", "ta_trend_indicators", "ta_smart_money_concepts", "inst_volume_profile"],
        context,
    )
    assert set(results) == {"ta_timeframe_analysis", "ta_trend_indicators", "ta_smart_money_concepts", "inst_volume_profile"}
    assert all(-1.0 <= result.score <= 1.0 for result in results.values())
    aggregate = GLOBAL_SKILL_ENGINE.aggregate(results)
    assert "aggregate_score" in aggregate
    assert aggregate["valid_count"] >= 3


def test_skill_powered_analysis_tool_and_base_agent_helpers():
    from python_brain.analysis_tools.trader_skill_stack import TraderSkillStackTool
    from python_brain.agents.base_agent import BaseAgent

    buffers = {"M5": _make_df(), "M15": _make_df(seed=2), "H1": _make_df(seed=3)}
    tool_result = TraderSkillStackTool().analyze(buffers, price=float(buffers["M15"]["close"].iloc[-1]), spread=1.0)
    assert tool_result.is_valid()
    assert "skill_valid_count" in tool_result.features
    assert "ta_trend_indicators__score" in tool_result.features

    class DummyAgent(BaseAgent):
        skill_role = "analyst"
        name = "dummy_agent"

        def run(self, context):
            return {}

    agent = DummyAgent()
    assert agent.available_skills(executable_only=True)
    one = agent.use_skill("ta_trend_indicators", {"buffers": buffers})
    assert one.is_valid()
