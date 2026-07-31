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


# ---------------------------------------------------------------- regression:
# every tool must compute (no silent exceptions) on plain pandas DataFrames,
# and the feature vector / aggregate score must never contain NaN.
def test_no_tool_silently_fails_on_plain_dataframes():
    from python_brain.confluence import ConfluenceEngine
    from python_brain.communication.data_parser import MarketSnapshot, TickData, AccountData
    from datetime import datetime, timezone
    from tests.test_smoke import _make_df

    buffers = {tf: _make_df(seed=i) for i, tf in enumerate(["M1", "M5", "M15", "M30", "H1", "H4", "D1"])}
    snap = MarketSnapshot(
        symbol="EURUSD", trigger="TICK",
        timestamp=datetime.now(timezone.utc),
        tick=TickData(1.0840, 1.0845, 1.5, 100),
        account=AccountData(10000, 10100, 9500, 500, 100, 100),
    )
    res = ConfluenceEngine().run(snap, buffers)

    failed = {name: r.errors for name, r in res["tool_results"].items() if r.errors}
    assert not failed, f"{len(failed)} tools raised exceptions: {list(failed)[:10]}"

    fv = res["feature_vector"]
    assert fv.ndim == 1
    assert not np.isnan(fv).any(), "feature vector contains NaN"
    assert np.isfinite(res["aggregate_score"]), "aggregate score is not finite"


def test_buffer_manager_returns_safe_frames():
    from python_brain.preprocessor import BufferManager
    from python_brain.preprocessor.buffer_manager import SafeDataFrame
    from tests.test_smoke import _make_df

    buf = BufferManager()
    buf.update({"M5": _make_df(seed=1), "H1": _make_df(seed=2)})
    for tf, df in buf.all().items():
        assert isinstance(df, SafeDataFrame)
    # boolean context must not raise
    assert bool(buf.get("M5"))
    assert not bool(buf.get("M1"))  # untouched TF stays empty
