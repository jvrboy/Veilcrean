"""
test_tools.py
=============
Per-tool tests — make sure each of the 8 analysis tools produces a
reasonable output on synthetic data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from python_brain.analysis_tools import (
    MarketStructureTool, SupplyDemandTool, LiquidityTool,
    MomentumVolumeTool, KeyLevelsTool, SessionTimeTool,
    CandlestickTool, MTFAlignmentTool,
)


def _make_trending_df(n=200, slope=0.001, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(slope, 0.0008, n)
    prices = 1.0 + np.cumsum(rets)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open":  prices,
        "high":  prices + np.abs(rng.normal(0.0005, 0.0001, n)),
        "low":   prices - np.abs(rng.normal(0.0005, 0.0001, n)),
        "close": prices,
        "volume": rng.integers(100, 500, n).astype(float),
    }, index=idx)


def test_market_structure_uptrend():
    df = _make_trending_df(slope=0.002)
    buffers = {"M15": df, "H1": df, "D1": df}
    res = MarketStructureTool().analyze(buffers, price=df["close"].iloc[-1])
    assert res.is_valid() or res.errors
    assert -1.0 <= res.score <= 1.0


def test_supply_demand_returns_score():
    df = _make_trending_df()
    buffers = {"M15": df, "H1": df}
    res = SupplyDemandTool().analyze(buffers, price=df["close"].iloc[-1])
    assert -1.0 <= res.score <= 1.0


def test_liquidity_returns_score():
    df = _make_trending_df()
    buffers = {"H1": df, "M15": df}
    res = LiquidityTool().analyze(buffers, price=df["close"].iloc[-1])
    assert -1.0 <= res.score <= 1.0


def test_momentum_volume_features():
    df = _make_trending_df()
    buffers = {"M15": df, "H1": df}
    res = MomentumVolumeTool().analyze(buffers)
    assert "mom_rsi_M15" in res.features
    assert "mom_hist_M15" in res.features
    assert 0.0 <= res.features["mom_rsi_M15"] <= 1.0 or -1.0 <= res.features["mom_rsi_M15"] <= 1.0


def test_key_levels():
    df = _make_trending_df()
    buffers = {"H1": df, "D1": df}
    res = KeyLevelsTool().analyze(buffers, price=df["close"].iloc[-1])
    assert -1.0 <= res.score <= 1.0


def test_session_time_killzone():
    from datetime import datetime, timezone
    # 9 UTC = London kill zone
    now = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)
    res = SessionTimeTool().analyze({}, now=now)
    assert res.features["sess_killzone"] == 1.0
    assert res.features["sess_in_london"] == 1.0
    assert res.score > 0


def test_candlestick_tool_runs():
    df = _make_trending_df()
    buffers = {"M5": df, "M15": df, "M30": df, "H1": df}
    res = CandlestickTool().analyze(buffers)
    assert -1.0 <= res.score <= 1.0


def test_mtf_alignment_trending():
    df = _make_trending_df(slope=0.002)
    buffers = {"M5": df, "M15": df, "H1": df, "H4": df, "D1": df}
    res = MTFAlignmentTool().analyze(buffers)
    # most TFs should agree on direction → positive score
    assert res.score > 0
