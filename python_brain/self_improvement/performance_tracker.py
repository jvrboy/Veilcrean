"""
performance_tracker.py
======================
Aggregates the journal into the metrics the bot — and you — care about.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .trade_journal import TradeJournal, TradeRecord


@dataclass
class PerformanceSnapshot:
    total_trades: int = 0
    wins:         int = 0
    losses:       int = 0
    win_rate:     float = 0.0
    profit_factor: float = 0.0
    avg_r:         float = 0.0
    avg_win_pnl:   float = 0.0
    avg_loss_pnl:  float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe:        float = 0.0
    by_regime:     Dict[str, Dict] = field(default_factory=dict)
    by_session:    Dict[str, Dict] = field(default_factory=dict)


class PerformanceTracker:
    def __init__(self, journal: TradeJournal):
        self.journal = journal

    # ------------------------------------------------------------------ API
    def snapshot(self) -> PerformanceSnapshot:
        closed = self.journal.all_closed()
        snap = PerformanceSnapshot()
        snap.total_trades = len(closed)
        if not closed:
            return snap

        pnls   = np.array([t.pnl for t in closed])
        wins   = pnls > 0
        losses = pnls <= 0
        snap.wins   = int(wins.sum())
        snap.losses = int(losses.sum())
        snap.win_rate = float(wins.mean()) if len(pnls) else 0.0

        gross_win  = float(pnls[wins].sum())  if wins.any()  else 0.0
        gross_loss = float(-pnls[losses].sum()) if losses.any() else 0.0
        snap.profit_factor = gross_win / max(gross_loss, 1e-9)

        snap.avg_r        = float(np.mean([t.r_achieved for t in closed]))
        snap.avg_win_pnl  = float(np.mean(pnls[wins]))  if wins.any()  else 0.0
        snap.avg_loss_pnl = float(np.mean(pnls[losses]))if losses.any() else 0.0

        # Drawdown over time
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        # Use absolute drawdown when peak is tiny (e.g. early trades)
        # Avoid div-by-zero or weird percentages when starting from 0
        peak_safe = np.where(np.abs(peak) < 1e-6, 1.0, peak)
        dd = (equity - peak) / np.abs(peak_safe)
        snap.max_drawdown_pct = float(dd.min() * 100.0)

        # Sharpe
        if pnls.std() > 0:
            snap.sharpe = float(pnls.mean() / pnls.std() * np.sqrt(252))

        snap.by_regime  = self._group(closed, key=lambda t: t.regime)
        snap.by_session = self._group(closed, key=lambda t: t.session)
        return snap

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _group(records: List[TradeRecord], key) -> Dict[str, Dict]:
        out: Dict[str, List[TradeRecord]] = defaultdict(list)
        for r in records:
            out[key(r) or "UNKNOWN"].append(r)
        agg = {}
        for k, rs in out.items():
            pnls = np.array([t.pnl for t in rs])
            wins = (pnls > 0).sum()
            agg[k] = {
                "n":     len(rs),
                "wins":  int(wins),
                "win_rate": float(wins / max(len(rs), 1)),
                "pnl_sum": float(pnls.sum()),
            }
        return agg
