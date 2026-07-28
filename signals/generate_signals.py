"""
generate_signals.py
====================
Batch entry point for the Veilcrean signal system.

    python -m signals.generate_signals            # scalp mode (default)
    python -m signals.generate_signals --mode swing
    python -m signals.generate_signals --close-manual   # mark open trades as
                                                        # manually closed first

Pipeline on every run:
  1. Load persistent learning state + historical ledger.
  2. (optional) Close any still-OPEN signals as MANUAL if the user closed them
     by hand, then score the rest against real price action (self-learning
     feedback -> updates per-instrument/per-mode parameters).
  3. For each of the 24 tracked instruments: fetch REAL data at the mode's
     timeframe, run the 144-tool confluence engine with that instrument's
     *learned* parameters, and emit BUY/SELL + ENTRY/TP/SL.
     Boom -> always BUY, Crash -> always SELL.
  4. Append new signals to the ledger and persist everything.
  5. Write latest_signals.json/.md and performance_report.md.

The tracked instrument universe is FIXED (data_feeds.INSTRUMENTS).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

from . import data_feeds, learning, signal_engine, tracker

HERE = os.path.dirname(os.path.abspath(__file__))
LATEST_JSON = os.path.join(HERE, "latest_signals.json")
LATEST_MD = os.path.join(HERE, "latest_signals.md")
REPORT_MD = os.path.join(HERE, "performance_report.md")

# how long a fresh signal stays valid before EXPIRED (hours)
EXPIRY_HOURS = {"scalp": 6, "swing": 24 * 14}


def _fmt(x):
    return f"{x:g}"


def run(mode: str = "scalp", close_manual: bool = False) -> dict:
    now = datetime.now(timezone.utc)
    state = learning.load_state()
    ledger = tracker.load_ledger()

    # ---- 1. house-keeping on prior signals ------------------------------- #
    manual_closed = tracker.close_open_manual(ledger, state) if close_manual else 0
    perf = tracker.evaluate_open_signals(ledger, state)

    # ---- 2. generate fresh signals --------------------------------------- #
    signals, errors = [], {}
    for name, meta in data_feeds.INSTRUMENTS.items():
        try:
            buffers, price = data_feeds.load_instrument(name, mode)
            params = learning.params_for(state, name, mode)
            signals.append(signal_engine.analyse(
                name, buffers, price, params, force=meta.get("force")))
        except Exception as e:      # noqa: BLE001
            errors[name] = str(e)

    # ---- 3. persist ------------------------------------------------------ #
    tracker.append_signals(ledger, signals, state, mode, EXPIRY_HOURS[mode])
    tracker.save_ledger(ledger)
    learning.save_state(state)

    payload = {
        "generated_at": now.isoformat(),
        "mode": mode,
        "universe_size": len(data_feeds.INSTRUMENTS),
        "manual_closed": manual_closed,
        "evaluation": perf,
        "errors": errors,
        "signals": signals,
    }
    with open(LATEST_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    _write_latest_md(payload)
    _write_report_md(state, ledger)
    return payload


def _write_latest_md(payload: dict) -> None:
    hold = "~minutes-hours" if payload["mode"] == "scalp" else "hours-days"
    lines = [
        f"# Veilcrean — Latest Signals ({payload['mode'].upper()})",
        "",
        f"*Generated: {payload['generated_at']}*  ",
        f"*Engine: 144-tool ConfluenceEngine · Instruments: {payload['universe_size']} "
        f"· Target hold: {hold}*",
        "",
        "> Educational / research output. Not financial advice. "
        "Boom = forced BUY, Crash = forced SELL per configuration.",
        "",
        "| Instrument | Signal | Entry | TP | SL | R:R | Conf | Conviction |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in payload["signals"]:
        lines.append(
            f"| {s['instrument']} | **{s['direction']}** | {_fmt(s['entry'])} | "
            f"{_fmt(s['tp'])} | {_fmt(s['sl'])} | {s['rr']} | "
            f"{s['confidence']} | {s['conviction']} |")
    ev = payload["evaluation"]
    lines += [
        "",
        "## Tracking feedback this run",
        f"- Manually closed (by user): **{payload['manual_closed']}**",
        f"- Open signals checked: **{ev['checked']}**",
        f"- Wins: **{ev['wins']}**  ·  Losses: **{ev['losses']}**  ·  "
        f"Expired: **{ev['expired']}**  ·  Still open: **{ev['still_open']}**",
    ]
    if payload["errors"]:
        lines += ["", "## Data errors", ""]
        lines += [f"- {k}: {v}" for k, v in payload["errors"].items()]
    with open(LATEST_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_report_md(state: dict, ledger: list) -> None:
    graded = [s for s in ledger if s.get("result") in ("WIN", "LOSS")]
    wins = sum(1 for s in graded if s["result"] == "WIN")
    total = len(graded)
    counts = {}
    for s in ledger:
        counts[s.get("result") or "OPEN"] = counts.get(s.get("result") or "OPEN", 0) + 1
    lines = [
        "# Veilcrean — Performance & Learning Report",
        "",
        f"*Updated: {datetime.now(timezone.utc).isoformat()}*",
        "",
        f"- Signals in ledger: **{len(ledger)}**",
        f"- Outcome breakdown: **{counts}**",
        f"- Graded (WIN/LOSS): **{total}**",
        (f"- Overall win rate: **{(wins / total * 100):.1f}%**" if total
         else "- Overall win rate: **n/a (no graded signals yet)**"),
        "",
        "## Per-instrument · per-mode learned parameters",
        "",
        "| Key | Signals | W | L | Win% | Threshold | ConfMult | R:R | k_SL |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(state.keys()):
        p = state[k]
        wr = f"{p['win_rate'] * 100:.0f}%" if p.get("win_rate") is not None else "—"
        lines.append(
            f"| {k} | {p['total_signals']} | {p['wins']} | {p['losses']} | "
            f"{wr} | {p['threshold']:.3f} | {p['conf_multiplier']:.2f} | "
            f"{p['rr']:.2f} | {p['k_sl']:.2f} |")
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scalp", "swing"], default="scalp")
    ap.add_argument("--close-manual", action="store_true",
                    help="mark all currently-open signals as manually closed")
    args = ap.parse_args()
    out = run(mode=args.mode, close_manual=args.close_manual)
    print(f"[{out['mode']}] generated {len(out['signals'])} signals "
          f"({len(out['errors'])} errors). manual_closed={out['manual_closed']} "
          f"eval={out['evaluation']}")
