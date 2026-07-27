"""
test_smoke.py
=============
Smoke tests — every import should succeed and the basic API should
work end-to-end on synthetic data.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------- import-only tests
def test_imports():
    """Every public module should import without error."""
    from python_brain import config
    from python_brain.communication   import ZMQServer, DataParser, MarketSnapshot
    from python_brain.preprocessor    import DataCleaner, Normalizer, BufferManager
    from python_brain.analysis_tools  import (
        MarketStructureTool, SupplyDemandTool, LiquidityTool,
        MomentumVolumeTool, KeyLevelsTool, SessionTimeTool,
        CandlestickTool, MTFAlignmentTool, ALL_TOOLS, ToolResult,
    )
    from python_brain.confluence      import FeatureBuilder, ConfluenceEngine
    from python_brain.neural_network  import (
        TradeDecisionNet, RiskManagementNet, RegimeClassifier,
        Trainer, Validator, ModelManager,
    )
    from python_brain.self_improvement import (
        TradeJournal, TradeRecord, PerformanceTracker, Retrainer, ThresholdAdjuster,
    )
    from python_brain.risk_management import PositionSizer, DrawdownGuard, ExposureManager
    from python_brain.utils            import get_logger, Alerter, Visualizer


# ---------------------------------------------------------------- helper
def _make_df(trend: float = 0.0, n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.001, n)
    prices = 1.0 + np.cumsum(rets)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open":  prices + rng.normal(0, 0.0002, n),
        "high":  prices + np.abs(rng.normal(0.0005, 0.0002, n)),
        "low":   prices - np.abs(rng.normal(0.0005, 0.0002, n)),
        "close": prices,
        "volume": rng.integers(100, 1000, n).astype(float),
    }, index=idx)


# ---------------------------------------------------------------- buffer
def test_buffer_update():
    from python_brain.preprocessor import BufferManager
    buf = BufferManager()
    df = _make_df()
    out = buf.update({"M5": df, "H1": df})
    assert "M5" in out
    assert "H1" in out
    assert len(out["M5"]) == len(df)


# ---------------------------------------------------------------- cleaner
def test_cleaner_drops_garbage():
    from python_brain.preprocessor import DataCleaner
    df = _make_df()
    df.iloc[0, df.columns.get_loc("high")] = -1  # garbage
    cl = DataCleaner()
    out = cl.clean(df)
    assert (out["high"] >= 0).all()


# ---------------------------------------------------------------- tools
def test_all_tools_return_results():
    from python_brain.confluence import ConfluenceEngine
    from python_brain.communication.data_parser import MarketSnapshot, TickData, AccountData
    from datetime import datetime, timezone

    buffers = {tf: _make_df(seed=i) for i, tf in enumerate(["M1","M5","M15","H1","H4","D1"])}
    snap = MarketSnapshot(
        symbol="EURUSD", trigger="TICK",
        timestamp=datetime.now(timezone.utc),
        tick=TickData(1.0840, 1.0845, 1.5, 100),
        account=AccountData(10000, 10100, 9500, 500, 100, 100),
    )
    eng = ConfluenceEngine()
    res = eng.run(snap, buffers)
    assert "feature_vector" in res
    assert res["feature_vector"].ndim == 1
    assert -1.0 <= res["aggregate_score"] <= 1.0


# ---------------------------------------------------------------- networks
def test_nets_forward():
    from python_brain.neural_network.models import (
        TradeDecisionNet, RiskManagementNet, RegimeClassifier,
    )
    import torch, torch.nn.functional as F
    x = torch.randn(4, 32)
    trade = TradeDecisionNet(32)
    risk  = RiskManagementNet(32)
    reg   = RegimeClassifier(32)
    logits, conf = trade(x)
    assert logits.shape == (4, 3)
    assert conf.shape == (4,)
    oh = F.one_hot(torch.tensor([0,1,2,0]), num_classes=3).float()
    sl, tp, lo = risk(x, oh, torch.tensor([0.5,0.6,0.7,0.8]))
    assert sl.shape == (4,1)
    out = reg(x)
    assert out.shape == (4, 5)


# ---------------------------------------------------------------- sizer
def test_position_sizer_respects_caps():
    from python_brain.risk_management import PositionSizer
    from python_brain.config import RISK_CFG
    lots = PositionSizer.lots(10000, sl_pips=20)
    assert RISK_CFG.lot_min <= lots <= RISK_CFG.lot_max


# ---------------------------------------------------------------- drawdown guard
def test_drawdown_guard_kill():
    from python_brain.risk_management import DrawdownGuard
    g = DrawdownGuard()
    g.update(10000)   # peak
    g.update(8900)    # 11% drawdown
    assert g.kill_switch


# ---------------------------------------------------------------- journal
def test_journal_roundtrip(tmp_path):
    from python_brain.self_improvement import TradeJournal, TradeRecord
    from python_brain.config import JOURNAL_DB
    import python_brain.config as cfg
    cfg.JOURNAL_DB = tmp_path / "j.db"
    j = TradeJournal(cfg.JOURNAL_DB)
    rec = TradeRecord(
        trade_id="t1", symbol="EURUSD", direction="BUY",
        opened_at=1000.0, entry_price=1.08, sl=1.07, tp=1.10,
        lots=0.1, confidence=0.8, regime="TRENDING",
    )
    j.open_trade(rec)
    j.close_trade("t1", exit_price=1.10, pnl=20.0, pnl_pct=2.0,
                  r_achieved=2.0, mae=5.0, mfe=25.0, is_win=1)
    closed = j.all_closed()
    assert len(closed) == 1
    assert closed[0].pnl == 20.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
