"""
force_retrain.py
================
Manually trigger a retraining cycle (bypasses the cadence gate).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python_brain.config import JOURNAL_DB
from python_brain.self_improvement import TradeJournal
from python_brain.neural_network  import ModelManager
from python_brain.utils            import get_logger

log = get_logger("retrain")


def main() -> int:
    j = TradeJournal(JOURNAL_DB)
    mm = ModelManager()
    n = j.count_closed()
    log.info(f"closed trades: {n}")
    if n < 50:
        log.warning("need at least 50 closed trades; aborting")
        return 1
    from python_brain.self_improvement import Retrainer
    # Use a tentative input dim; the real one is set on first train.
    rt = Retrainer(j, mm, input_dim=64)
    res = rt.run()
    log.info(f"result: {res}")
    return 0 if res.deployed else 2


if __name__ == "__main__":
    sys.exit(main())
