"""
learning.py
===========
Per-instrument self-learning state. This is the memory that makes the repo
improve over time.

For every instrument we persist adaptive trade parameters and a running
performance record. After each batch of signals is scored by the tracker,
`update_from_outcome` nudges the parameters:

    * win_rate low   -> be MORE selective (raise threshold, cut confidence
                        multiplier) and demand a better reward-to-risk.
    * win_rate high  -> be MORE aggressive (lower threshold, raise multiplier).
    * repeated SL hits -> widen the SL multiple (k_sl); repeated TP hits with
                          room to spare -> tighten it.

State lives in signals/learning_state.json and is committed to the repo, so
learning survives across runs and machines.
"""
from __future__ import annotations

import json
import os
from typing import Dict

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "learning_state.json")

DEFAULT_PARAMS = {
    "threshold": 0.04,       # min |adj_score| for a NORMAL directional call
    "conf_multiplier": 1.0,  # scales the raw aggregate score
    "rr": 2.0,               # reward : risk
    "k_sl": 1.5,             # SL distance = k_sl * ATR
    "wins": 0,
    "losses": 0,
    "open": 0,
    "total_signals": 0,
    "win_rate": None,
    "sl_hits": 0,
    "tp_hits": 0,
    "last_updated": None,
}

# hard bounds so the learner can never drift into nonsense
BOUNDS = {
    "threshold": (0.01, 0.25),
    "conf_multiplier": (0.4, 2.0),
    "rr": (1.2, 4.0),
    "k_sl": (0.8, 4.0),
}


def _clip(key: str, val: float) -> float:
    lo, hi = BOUNDS[key]
    return max(lo, min(hi, val))


def load_state() -> Dict[str, dict]:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state: Dict[str, dict]) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def params_for(state: Dict[str, dict], instrument: str) -> dict:
    """Return current adaptive params for an instrument (creating defaults)."""
    if instrument not in state:
        state[instrument] = dict(DEFAULT_PARAMS)
    # backfill any newly-added keys on old state files
    for k, v in DEFAULT_PARAMS.items():
        state[instrument].setdefault(k, v)
    return state[instrument]


def register_open(state: Dict[str, dict], instrument: str) -> None:
    p = params_for(state, instrument)
    p["open"] += 1
    p["total_signals"] += 1


def update_from_outcome(state: Dict[str, dict], instrument: str,
                        outcome: str, hit: str, now_iso: str) -> None:
    """
    Fold a scored signal back into the instrument's parameters.

    outcome : "WIN" | "LOSS"
    hit     : "TP" | "SL" | "NONE"
    """
    p = params_for(state, instrument)
    p["open"] = max(0, p["open"] - 1)

    if outcome == "WIN":
        p["wins"] += 1
        if hit == "TP":
            p["tp_hits"] += 1
    elif outcome == "LOSS":
        p["losses"] += 1
        if hit == "SL":
            p["sl_hits"] += 1

    graded = p["wins"] + p["losses"]
    if graded > 0:
        p["win_rate"] = round(p["wins"] / graded, 4)

    # only adapt once we have a minimum sample so we don't overfit noise
    if graded >= 4:
        wr = p["win_rate"]
        if wr < 0.40:
            p["threshold"] = _clip("threshold", p["threshold"] * 1.15)
            p["conf_multiplier"] = _clip("conf_multiplier", p["conf_multiplier"] * 0.92)
            p["rr"] = _clip("rr", p["rr"] + 0.10)
        elif wr > 0.60:
            p["threshold"] = _clip("threshold", p["threshold"] * 0.92)
            p["conf_multiplier"] = _clip("conf_multiplier", p["conf_multiplier"] * 1.06)

        # SL sizing: too many stop-outs -> give trades more room
        if p["sl_hits"] >= 3 and p["sl_hits"] > p["tp_hits"]:
            p["k_sl"] = _clip("k_sl", p["k_sl"] * 1.10)
            p["sl_hits"] = 0  # reset the counter after acting
        elif p["tp_hits"] >= 3 and p["tp_hits"] > p["sl_hits"]:
            p["k_sl"] = _clip("k_sl", p["k_sl"] * 0.95)
            p["tp_hits"] = 0

    p["last_updated"] = now_iso
