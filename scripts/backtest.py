"""
backtest.py
===========
Offline backtest of the confluence + neural-network decision pipeline
on historical (or synthetic) OHLCV data.

Usage:
    python scripts/backtest.py --symbol EURUSD --bars 5000 --n-trades 100
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from python_brain.communication.data_parser import MarketSnapshot, TickData, AccountData
from python_brain.preprocessor              import BufferManager
from python_brain.confluence                import ConfluenceEngine
from python_brain.neural_network            import TradeDecisionNet, RiskManagementNet, RegimeClassifier
from python_brain.utils                     import get_logger, Visualizer

log = get_logger("backtest")


def make_synthetic(n: int, slope: float = 0.0, vol: float = 0.001, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(slope, vol, n)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",  default="EURUSD")
    ap.add_argument("--bars",    type=int, default=2000)
    ap.add_argument("--n-trades", type=int, default=50)
    ap.add_argument("--sl-pips", type=float, default=20.0)
    ap.add_argument("--tp-pips", type=float, default=40.0)
    args = ap.parse_args()

    log.info(f"backtest: {args.symbol} | {args.bars} bars | target {args.n_trades} trades")

    # 1. build synthetic data for all TFs
    pip = 0.0001
    base = make_synthetic(args.bars, seed=42)
    buffers = BufferManager()
    for tf, s in zip(["M1","M5","M15","M30","H1","H4","D1"], range(7)):
        buffers.update({tf: make_synthetic(args.bars, seed=s + 100)})
    log.info("buffers built")

    # 2. decision network (random untrained)
    eng = ConfluenceEngine()
    trade_net = TradeDecisionNet(64)   # placeholder; rebuilt with real dim on first pass
    trade_net.eval()                   # eval mode (BatchNorm needs >1 sample when training)

    # 3. walk the bar history
    eq = 10000.0
    eq_history = []
    trade_log = []
    in_pos = None   # {'side', 'entry', 'sl', 'tp'}

    for i in range(60, args.bars):
        snap = MarketSnapshot(
            symbol=args.symbol, trigger="BAR",
            timestamp=base.index[i].to_pydatetime(),
            tick=TickData(base["close"].iloc[i] - 0.0001,
                         base["close"].iloc[i] + 0.0001, 1.5, 100),
            account=AccountData(eq, eq, eq - 500, 500, 0, 100),
        )
        result = eng.run(snap, buffers.all())
        fv = result["feature_vector"]
        # lazy-init network with actual feature dim
        if trade_net.backbone[0].in_features != fv.shape[0]:
            trade_net = TradeDecisionNet(fv.shape[0])
            trade_net.eval()
        import torch
        with torch.no_grad():
            logits, conf = trade_net(torch.tensor(fv, dtype=torch.float32).unsqueeze(0))
        import torch.nn.functional as F
        probs = F.softmax(logits, dim=-1)[0].numpy()
        action_idx = int(np.argmax(probs))
        action = ["BUY", "SELL", "HOLD"][action_idx]
        price = float(base["close"].iloc[i])

        # simple exit logic if in position
        if in_pos is not None:
            hi, lo = base["high"].iloc[i], base["low"].iloc[i]
            if in_pos["side"] == "BUY":
                if lo <= in_pos["sl"]: pnl = -(args.sl_pips * pip) * 100000
                elif hi >= in_pos["tp"]: pnl = (args.tp_pips * pip) * 100000
                else: pnl = 0
            else:
                if hi >= in_pos["sl"]: pnl = -(args.sl_pips * pip) * 100000
                elif lo <= in_pos["tp"]: pnl = (args.tp_pips * pip) * 100000
                else: pnl = 0
            if pnl != 0:
                eq += pnl / 100   # 0.01 lot
                trade_log.append(pnl / 100)
                eq_history.append(eq)
                in_pos = None
                if len(trade_log) >= args.n_trades:
                    break

        # entry logic
        if in_pos is None and action != "HOLD" and conf.item() > 0.55:
            sl = price - args.sl_pips * pip if action == "BUY" else price + args.sl_pips * pip
            tp = price + args.tp_pips * pip if action == "BUY" else price - args.tp_pips * pip
            in_pos = {"side": action, "entry": price, "sl": sl, "tp": tp}

    # 4. report
    if not trade_log:
        log.warning("no trades generated — try lowering confidence or shortening sl/tp")
        return 0
    pnls = np.array(trade_log)
    wins = (pnls > 0).sum()
    log.info(f"trades   : {len(pnls)}")
    log.info(f"wins     : {wins}  ({wins/len(pnls):.1%})")
    log.info(f"net pnl  : ${pnls.sum():.2f}")
    log.info(f"avg pnl  : ${pnls.mean():.2f}")
    log.info(f"max dd   : ${(np.maximum.accumulate(np.cumsum(pnls)) - np.cumsum(pnls)).max():.2f}")

    # 5. charts
    vis = Visualizer()
    p_eq  = vis.equity_curve(pnls.tolist(), name=f"backtest_{args.symbol}_equity.png")
    p_dd  = vis.drawdown(pnls.tolist(),   name=f"backtest_{args.symbol}_dd.png")
    log.info(f"equity chart: {p_eq}")
    log.info(f"dd    chart: {p_dd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
