"""
learning.py
===========
Per-instrument, per-mode self-learning state. This is the memory that makes the
repo improve over time.

State is keyed by "<instrument>::<mode>" (e.g. "EURUSD::scalp") so scalping and
swing strategies learn independently. After each batch of signals is scored by
the tracker, `update_from_outcome` nudges the parameters:

    * win_rate low   -> be MORE selective (raise threshold, cut confidence
                        multiplier) and demand a better reward-to-risk.
    * win_rate high  -> be MORE aggressive (lower threshold, raise multiplier).
    * repeated SL hits -> widen the SL multiple (k_sl); repeated TP hits ->
                          tighten it.

State lives in signals/learning_state.json and is committed to the repo.
"""
from __future__ import annotations

import json
import os
from typing import Dict

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "learning_state.json")

# mode-specific starting points
MODE_DEFAULTS = {
    "swing": {"threshold": 0.04, "conf_multiplier": 1.0, "rr": 2.0, "k_sl": 1.5},
    "scalp": {"threshold": 0.03, "conf_multiplier": 1.0, "rr": 1.5, "k_sl": 0.7},
}

_BASE = {
    "wins": 0, "losses": 0, "open": 0, "total_signals": 0,
    "win_rate": None, "sl_hits": 0, "tp_hits": 0, "last_updated": None,
    "mode": None,
}

BOUNDS = {
    "threshold": (0.01, 0.25),
    "conf_multiplier": (0.4, 2.0),
    "rr": (1.0, 4.0),
    "k_sl": (0.3, 4.0),
}


def key(instrument: str, mode: str) -> str:
    return f"{instrument}::{mode}"


def _clip(k: str, v: float) -> float:
    lo, hi = BOUNDS[k]
    return max(lo, min(hi, v))


def load_state() -> Dict[str, dict]:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state: Dict[str, dict]) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def params_for(state: Dict[str, dict], instrument: str, mode: str) -> dict:
    """Return current adaptive params for an instrument+mode (seeding defaults)."""
    k = key(instrument, mode)
    if k not in state:
        p = dict(_BASE)
        p.update(MODE_DEFAULTS.get(mode, MODE_DEFAULTS["swing"]))
        p["mode"] = mode
        state[k] = p
    for kk, vv in _BASE.items():
        state[k].setdefault(kk, vv)
    return state[k]


def register_open(state: Dict[str, dict], instrument: str, mode: str) -> None:
    p = params_for(state, instrument, mode)
    p["open"] += 1
    p["total_signals"] += 1


def update_from_outcome(state: Dict[str, dict], instrument: str, mode: str,
                        outcome: str, hit: str, now_iso: str) -> None:
    """Fold a scored signal back into the instrument's parameters.

    outcome : "WIN" | "LOSS" | "EXPIRED" | "MANUAL"
    hit     : "TP" | "SL" | "NONE"
    """
    p = params_for(state, instrument, mode)
    p["open"] = max(0, p["open"] - 1)

    if outcome == "WIN":
        p["wins"] += 1
        if hit == "TP":
            p["tp_hits"] += 1
    elif outcome == "LOSS":
        p["losses"] += 1
        if hit == "SL":
            p["sl_hits"] += 1
    # EXPIRED / MANUAL: counted as neither win nor loss, only frees the slot

    graded = p["wins"] + p["losses"]
    if graded > 0:
        p["win_rate"] = round(p["wins"] / graded, 4)

    if graded >= 4:
        wr = p["win_rate"]
        if wr < 0.40:
            p["threshold"] = _clip("threshold", p["threshold"] * 1.15)
            p["conf_multiplier"] = _clip("conf_multiplier", p["conf_multiplier"] * 0.92)
            p["rr"] = _clip("rr", p["rr"] + 0.10)
        elif wr > 0.60:
            p["threshold"] = _clip("threshold", p["threshold"] * 0.92)
            p["conf_multiplier"] = _clip("conf_multiplier", p["conf_multiplier"] * 1.06)

        if p["sl_hits"] >= 3 and p["sl_hits"] > p["tp_hits"]:
            p["k_sl"] = _clip("k_sl", p["k_sl"] * 1.10)
            p["sl_hits"] = 0
        elif p["tp_hits"] >= 3 and p["tp_hits"] > p["sl_hits"]:
            p["k_sl"] = _clip("k_sl", p["k_sl"] * 0.95)
            p["tp_hits"] = 0

    p["last_updated"] = now_iso
