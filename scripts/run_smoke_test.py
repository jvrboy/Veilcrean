"""
run_smoke_test.py
=================
End-to-end smoke test that:
  1. Generates synthetic candle data
  2. Builds a fake MarketSnapshot
  3. Runs the full pipeline (preprocess → tools → feature vector → decision)
  4. Verifies a decision comes out

Useful as a CI-style check that everything imports & runs.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from python_brain.communication.data_parser import MarketSnapshot, TickData, AccountData
from python_brain.preprocessor              import BufferManager
from python_brain.confluence                import ConfluenceEngine
from python_brain.neural_network            import TradeDecisionNet, ModelManager
from python_brain.utils                     import get_logger

log = get_logger("smoke")


def make_df(n: int = 300, slope: float = 0.001, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(slope, 0.0008, n)
    prices = 1.0 + np.cumsum(rets)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open":   prices,
        "high":   prices + np.abs(rng.normal(0.0005, 0.0002, n)),
        "low":    prices - np.abs(rng.normal(0.0005, 0.0002, n)),
        "close":  prices,
        "volume": rng.integers(100, 500, n).astype(float),
    }, index=idx)


def main() -> int:
    log.info("=" * 60)
    log.info("  Veilcrean — End-to-end Smoke Test")
    log.info("=" * 60)

    # 1. buffers
    buf = BufferManager()
    for tf, s in zip(["M1","M5","M15","M30","H1","H4","D1"], range(7)):
        buf.update({tf: make_df(seed=s)})
    buffers = buf.all()
    log.info(f"built {len(buffers)} TF buffers: {list(buffers.keys())}")

    # 2. snapshot
    snap = MarketSnapshot(
        symbol="EURUSD", trigger="TICK",
        timestamp=datetime.now(timezone.utc),
        tick=TickData(1.0840, 1.0845, 1.5, 200),
        account=AccountData(10000, 10100, 9500, 500, 100, 100),
    )

    # 3. confluence
    eng = ConfluenceEngine()
    result = eng.run(snap, buffers)
    log.info(f"feature vector dim = {result['feature_vector'].shape[0]}")
    log.info(f"aggregate score    = {result['aggregate_score']:.3f}")
    for name, r in result["tool_results"].items():
        log.info(f"  {name:22s} score={r.score:+.3f}  conf={r.confidence:.2f}")

    # 4. network decision (random untrained net)
    import torch
    net = TradeDecisionNet(result["feature_vector"].shape[0])
    net.eval()  # use eval mode (BatchNorm needs >1 sample in training mode)
    x = torch.tensor(result["feature_vector"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, conf = net(x)
    import torch.nn.functional as F
    probs = F.softmax(logits, dim=-1)[0].numpy()
    action = ["BUY", "SELL", "HOLD"][int(np.argmax(probs))]
    log.info(f"decision: {action} (BUY={probs[0]:.2f} SELL={probs[1]:.2f} HOLD={probs[2]:.2f}) conf={conf.item():.2f}")

    log.info("=" * 60)
    log.info("  ✅ Smoke test passed")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
