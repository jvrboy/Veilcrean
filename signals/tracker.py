"""
tracker.py
==========
Signal ledger + performance evaluator.

Every generated signal is appended to signals/ledger.json with status OPEN.
On each run, `evaluate_open_signals` replays real price action since a signal
was issued and decides whether TP or SL was touched first, marking the signal
WIN / LOSS / EXPIRED and feeding the result back into the learning layer.

This is the closed loop that lets the repo learn from its own history.
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

# signals older than this (hours) with no TP/SL touch are marked EXPIRED
EXPIRY_HOURS = 24 * 14


def load_ledger() -> List[dict]:
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return []


def save_ledger(ledger: List[dict]) -> None:
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def append_signals(ledger: List[dict], signals: List[dict],
                   state: dict) -> None:
    now = datetime.now(timezone.utc)
    for sig in signals:
        entry = dict(sig)
        entry["status"] = "OPEN"
        entry["issued_at"] = now.isoformat()
        entry["issued_epoch"] = int(now.timestamp())
        entry["id"] = f"{sig['instrument'].replace(' ', '_')}-{int(now.timestamp())}"
        entry["result"] = None
        entry["hit"] = None
        entry["closed_at"] = None
        ledger.append(entry)
        learning.register_open(state, sig["instrument"])


def _touch_first(path, entry, tp, sl, direction):
    """Walk candles and report which of TP/SL was hit first ('TP','SL',None)."""
    for _, row in path.iterrows():
        hi, lo = float(row["high"]), float(row["low"])
        if direction == "BUY":
            hit_tp, hit_sl = hi >= tp, lo <= sl
        else:
            hit_tp, hit_sl = lo <= tp, hi >= sl
        if hit_tp and hit_sl:
            # both in same candle -> assume SL first (conservative)
            return "SL"
        if hit_tp:
            return "TP"
        if hit_sl:
            return "SL"
    return None


def evaluate_open_signals(ledger: List[dict], state: dict) -> dict:
    """Score every OPEN signal against real price action. Returns a summary."""
    now = datetime.now(timezone.utc)
    summary = {"checked": 0, "wins": 0, "losses": 0, "expired": 0, "still_open": 0}

    # cache price paths per instrument to avoid duplicate network calls
    path_cache: dict = {}

    for sig in ledger:
        if sig.get("status") != "OPEN":
            continue
        summary["checked"] += 1
        inst = sig["instrument"]
        try:
            if inst not in path_cache:
                path_cache[inst] = data_feeds.price_path_since(
                    inst, sig["issued_epoch"])
            path = path_cache[inst]
        except Exception:      # noqa: BLE001
            summary["still_open"] += 1
            continue

        hit = _touch_first(path, sig, sig["tp"], sig["sl"], sig["direction"]) \
            if path is not None and len(path) else None

        if hit == "TP":
            sig.update(status="CLOSED", result="WIN", hit="TP",
                       closed_at=now.isoformat())
            learning.update_from_outcome(state, inst, "WIN", "TP", now.isoformat())
            summary["wins"] += 1
        elif hit == "SL":
            sig.update(status="CLOSED", result="LOSS", hit="SL",
                       closed_at=now.isoformat())
            learning.update_from_outcome(state, inst, "LOSS", "SL", now.isoformat())
            summary["losses"] += 1
        else:
            age_h = (now.timestamp() - sig["issued_epoch"]) / 3600.0
            if age_h > EXPIRY_HOURS:
                sig.update(status="CLOSED", result="EXPIRED", hit="NONE",
                           closed_at=now.isoformat())
                learning.update_from_outcome(state, inst, "EXPIRED", "NONE",
                                             now.isoformat())
                summary["expired"] += 1
            else:
                summary["still_open"] += 1
    return summary
