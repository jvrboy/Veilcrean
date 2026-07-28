"""
generate_signals.py
====================
Batch entry point for the Veilcrean signal system. Run:

    python -m signals.generate_signals

Pipeline on every run:
  1. Load the persistent learning state and the historical ledger.
  2. Score all previously-OPEN signals against real price action (self-learning
     feedback -> updates per-instrument parameters).
  3. For each of the 24 tracked instruments: fetch REAL data, run the 144-tool
     confluence engine using that instrument's *learned* parameters, and emit a
     BUY/SELL signal with ENTRY / TP / SL.  Boom -> always BUY, Crash -> SELL.
  4. Append the new signals to the ledger and persist everything.
  5. Write latest_signals.json / .md and performance_report.md.

The tracked instrument universe is FIXED (see data_feeds.INSTRUMENTS) so the
repo always analyses exactly these markets and only these.
"""
from __future__ import annotations

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


def _fmt(x):
    return f"{x:g}"


def run() -> dict:
    now = datetime.now(timezone.utc)
    state = learning.load_state()
    ledger = tracker.load_ledger()

    # ---- 1. learn from history ------------------------------------------- #
    perf = tracker.evaluate_open_signals(ledger, state)

    # ---- 2. generate fresh signals --------------------------------------- #
    signals = []
    errors = {}
    for name, meta in data_feeds.INSTRUMENTS.items():
        try:
            buffers, price = data_feeds.load_instrument(name)
            params = learning.params_for(state, name)
            sig = signal_engine.analyse(name, buffers, price, params,
                                        force=meta.get("force"))
            signals.append(sig)
        except Exception as e:      # noqa: BLE001
            errors[name] = str(e)

    # ---- 3. persist ------------------------------------------------------ #
    tracker.append_signals(ledger, signals, state)
    tracker.save_ledger(ledger)
    learning.save_state(state)

    payload = {
        "generated_at": now.isoformat(),
        "universe_size": len(data_feeds.INSTRUMENTS),
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
    ts = payload["generated_at"]
    lines = [
        "# Veilcrean — Latest Signals",
        "",
        f"*Generated: {ts}*  ",
        f"*Engine: 144-tool ConfluenceEngine · Instruments: {payload['universe_size']}*",
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
        "## Learning feedback this run",
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
    lines = [
        "# Veilcrean — Performance & Learning Report",
        "",
        f"*Updated: {datetime.now(timezone.utc).isoformat()}*",
        "",
        f"- Signals in ledger: **{len(ledger)}**",
        f"- Graded (WIN/LOSS): **{total}**",
        f"- Overall win rate: **{(wins / total * 100):.1f}%**" if total else
        "- Overall win rate: **n/a (no graded signals yet)**",
        "",
        "## Per-instrument learned parameters",
        "",
        "| Instrument | Signals | W | L | Win% | Threshold | ConfMult | R:R | k_SL |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for inst in sorted(state.keys()):
        p = state[inst]
        wr = f"{p['win_rate'] * 100:.0f}%" if p.get("win_rate") is not None else "—"
        lines.append(
            f"| {inst} | {p['total_signals']} | {p['wins']} | {p['losses']} | "
            f"{wr} | {p['threshold']:.3f} | {p['conf_multiplier']:.2f} | "
            f"{p['rr']:.2f} | {p['k_sl']:.2f} |")
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    out = run()
    print(f"Generated {len(out['signals'])} signals "
          f"({len(out['errors'])} errors). "
          f"Evaluation: {out['evaluation']}")
