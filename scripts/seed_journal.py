"""
seed_journal.py
===============
Seed the trade journal with N synthetic closed trades. Useful for
testing the retraining pipeline before you've accumulated any real
trades.

Usage:
    python scripts/seed_journal.py --n 200
"""
from __future__ import annotations
import argparse
import random
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from python_brain.config import JOURNAL_DB
from python_brain.self_improvement import TradeJournal, TradeRecord
from python_brain.utils import get_logger

log = get_logger("seed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--win-rate", type=float, default=0.55)
    ap.add_argument("--feature-dim", type=int, default=64)
    args = ap.parse_args()

    j = TradeJournal(JOURNAL_DB)
    rng = np.random.default_rng(0)
    regimes = ["TRENDING", "RANGING", "VOLATILE", "CHOPPY", "BREAKOUT"]
    sessions = ["asian", "london", "newyork", "overlap"]

    n = args.n
    n_wins = int(n * args.win_rate)
    outcomes = [1]*n_wins + [0]*(n - n_wins)
    random.shuffle(outcomes)

    for i, win in enumerate(outcomes):
        direction = random.choice(["BUY", "SELL"])
        pnl = float(rng.normal(40, 25)) if win else float(rng.normal(-20, 15))
        entry = 1.0800 + rng.normal(0, 0.005)
        sl = entry - 0.0020 if direction == "BUY" else entry + 0.0020
        tp = entry + 0.0040 if direction == "BUY" else entry - 0.0040
        exit_price = tp if win else sl
        rec = TradeRecord(
            trade_id=str(uuid.uuid4())[:12],
            symbol="EURUSD",
            direction=direction,
            opened_at=time.time() - (n - i) * 3600,
            closed_at=time.time() - (n - i) * 3600 + 1800,
            entry_price=float(entry),
            exit_price=float(exit_price),
            sl=float(sl),
            tp=float(tp),
            lots=0.10,
            pnl=pnl,
            pnl_pct=pnl / 100,
            r_achieved=2.0 if win else -1.0,
            confidence=float(rng.uniform(0.55, 0.95)),
            regime=random.choice(regimes),
            session=random.choice(sessions),
            weekday=random.randint(0, 4),
            strategy_tag="seed",
            feature_vec=list(rng.normal(0, 1, args.feature_dim).astype(float)),
            mae=float(rng.uniform(5, 30)),
            mfe=float(rng.uniform(10, 60)),
            is_win=win,
            notes="seeded",
        )
        j.open_trade(rec)
        j.close_trade(rec.trade_id, rec.exit_price, rec.pnl, rec.pnl_pct,
                      rec.r_achieved, rec.mae, rec.mfe, rec.is_win, rec.notes)
    log.info(f"seeded {n} trades (target WR={args.win_rate:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
