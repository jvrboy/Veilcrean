"""
tracker.py
==========
Signal ledger + performance evaluator.

Every generated signal is appended to signals/ledger.json with status OPEN and
its own expiry window. On each run:
  * `close_open_manual` records signals the user closed by hand (neither win nor
    loss) so the history stays truthful.
  * `evaluate_open_signals` replays real price action since a signal was issued
    and decides whether TP or SL was touched first, marking it WIN / LOSS /
    EXPIRED and feeding the result back into the learning layer.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List

from . import data_feeds
from . import learning

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ledger.json")

# default expiry per mode (hours) if a signal didn't store its own
DEFAULT_EXPIRY = {"scalp": 6, "swing": 24 * 14}


def load_ledger() -> List[dict]:
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return []


def save_ledger(ledger: List[dict]) -> None:
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def append_signals(ledger: List[dict], signals: List[dict], state: dict,
                   mode: str, expiry_hours: float) -> None:
    now = datetime.now(timezone.utc)
    for sig in signals:
        e = dict(sig)
        e["mode"] = mode
        e["status"] = "OPEN"
        e["issued_at"] = now.isoformat()
        e["issued_epoch"] = int(now.timestamp())
        e["expiry_hours"] = expiry_hours
        e["learn_key"] = learning.key(sig["instrument"], mode)
        e["id"] = f"{sig['instrument'].replace(' ', '_')}-{mode}-{int(now.timestamp())}"
        e["result"] = None
        e["hit"] = None
        e["closed_at"] = None
        ledger.append(e)
        learning.register_open(state, sig["instrument"], mode)


def close_open_manual(ledger: List[dict], state: dict) -> int:
    """Mark all currently-OPEN signals as manually closed by the user."""
    now = datetime.now(timezone.utc)
    n = 0
    for sig in ledger:
        if sig.get("status") != "OPEN":
            continue
        mode = sig.get("mode", "swing")
        sig.update(status="CLOSED", result="MANUAL", hit="NONE",
                   closed_at=now.isoformat(),
                   note="closed manually by user")
        learning.update_from_outcome(state, sig["instrument"], mode,
                                     "MANUAL", "NONE", now.isoformat())
        n += 1
    return n


def _touch_first(path, tp, sl, direction):
    for _, row in path.iterrows():
        hi, lo = float(row["high"]), float(row["low"])
        if direction == "BUY":
            hit_tp, hit_sl = hi >= tp, lo <= sl
        else:
            hit_tp, hit_sl = lo <= tp, hi >= sl
        if hit_tp and hit_sl:
            return "SL"           # both in one candle -> conservative
        if hit_tp:
            return "TP"
        if hit_sl:
            return "SL"
    return None


def evaluate_open_signals(ledger: List[dict], state: dict) -> dict:
    now = datetime.now(timezone.utc)
    summary = {"checked": 0, "wins": 0, "losses": 0, "expired": 0, "still_open": 0}
    path_cache: dict = {}

    for sig in ledger:
        if sig.get("status") != "OPEN":
            continue
        summary["checked"] += 1
        inst = sig["instrument"]
        mode = sig.get("mode", "swing")
        ck = (inst, mode)
        try:
            if ck not in path_cache:
                path_cache[ck] = data_feeds.price_path_since(
                    inst, sig["issued_epoch"], mode)
            path = path_cache[ck]
        except Exception:      # noqa: BLE001
            summary["still_open"] += 1
            continue

        hit = _touch_first(path, sig["tp"], sig["sl"], sig["direction"]) \
            if path is not None and len(path) else None

        if hit == "TP":
            sig.update(status="CLOSED", result="WIN", hit="TP",
                       closed_at=now.isoformat())
            learning.update_from_outcome(state, inst, mode, "WIN", "TP",
                                         now.isoformat())
            summary["wins"] += 1
        elif hit == "SL":
            sig.update(status="CLOSED", result="LOSS", hit="SL",
                       closed_at=now.isoformat())
            learning.update_from_outcome(state, inst, mode, "LOSS", "SL",
                                         now.isoformat())
            summary["losses"] += 1
        else:
            age_h = (now.timestamp() - sig["issued_epoch"]) / 3600.0
            if age_h > sig.get("expiry_hours", DEFAULT_EXPIRY.get(mode, 24)):
                sig.update(status="CLOSED", result="EXPIRED", hit="NONE",
                           closed_at=now.isoformat())
                learning.update_from_outcome(state, inst, mode, "EXPIRED",
                                             "NONE", now.isoformat())
                summary["expired"] += 1
            else:
                summary["still_open"] += 1
    return summary
