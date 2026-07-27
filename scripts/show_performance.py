"""
show_performance.py
===================
Print a summary of the trade journal's current performance.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python_brain.config import JOURNAL_DB
from python_brain.self_improvement import TradeJournal, PerformanceTracker
from python_brain.utils import get_logger, Visualizer

log = get_logger("perf")


def main() -> int:
    j = TradeJournal(JOURNAL_DB)
    pt = PerformanceTracker(j)
    snap = pt.snapshot()

    log.info("=" * 60)
    log.info("  Veilcrean — Performance Summary")
    log.info("=" * 60)
    log.info(f"Total trades   : {snap.total_trades}")
    log.info(f"Wins / Losses  : {snap.wins} / {snap.losses}")
    log.info(f"Win rate       : {snap.win_rate:.1%}")
    log.info(f"Profit factor  : {snap.profit_factor:.2f}")
    log.info(f"Avg R          : {snap.avg_r:.2f}")
    log.info(f"Avg win pnl    : ${snap.avg_win_pnl:.2f}")
    log.info(f"Avg loss pnl   : ${snap.avg_loss_pnl:.2f}")
    log.info(f"Max drawdown   : {snap.max_drawdown_pct:.2f}%")
    log.info(f"Sharpe         : {snap.sharpe:.2f}")
    if snap.by_regime:
        log.info("")
        log.info("By regime:")
        for r, m in snap.by_regime.items():
            log.info(f"  {r:10s}  n={m['n']:3d}  WR={m['win_rate']:.1%}  pnl=${m['pnl_sum']:.2f}")
    if snap.by_session:
        log.info("")
        log.info("By session:")
        for r, m in snap.by_session.items():
            log.info(f"  {r:10s}  n={m['n']:3d}  WR={m['win_rate']:.1%}  pnl=${m['pnl_sum']:.2f}")

    # generate charts
    if snap.total_trades > 5:
        closed = j.all_closed()
        pnls = [t.pnl for t in closed]
        vis = Visualizer()
        vis.equity_curve(pnls, name="current_equity.png")
        vis.drawdown(pnls, name="current_drawdown.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
