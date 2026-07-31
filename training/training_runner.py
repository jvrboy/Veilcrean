"""
Training runner — orchestrates the full forward-test training.

For each instrument, for each timeframe:
1. Fetches maximum historical candle data from Deriv
2. Walks forward candle-by-candle (oldest to newest)
3. At each candle, generates a signal using all indicators
4. Simulates the trade on future candles
5. Learns from the outcome (failure analysis + Q-learning threshold update)
6. Applies learned patterns to filter future signals
7. Applies learned TP/SL adjustments to future signals
8. Saves all data to JSON files

Across RUNS (default 15):
- The LearningEngine state (Q-table, failure patterns, TP/SL adjustments)
  is carried forward from one run to the next, so each run literally
  improves on the previous run's mistakes.
- Per-run win/loss counts are logged so you can watch the win rate climb.

The agent improves per signal: each failure teaches it what to avoid next time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, UTC
from typing import Optional

import numpy as np

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.deriv_client import DerivClient, get_all_instruments, get_all_timeframes, TIMEFRAMES
from training.signal_generator import generate_signal, Signal
from training.trade_simulator import simulate_trade, TradeOutcome
from training.learning_engine import LearningEngine


# Supabase config (read from env) — optional; JSON is the primary store
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

# Pip sizes per market
PIP_SIZES = {
    "forex": 0.0001,
    "commodity": 0.01,    # metals
    "cryptocurrency": 1.0,  # crypto
    "synthetic_index": 0.001,  # synthetics
}

# Training config
MIN_HISTORY = 50          # min candles before generating signals
MAX_HOLD_BARS = 20        # max candles to hold a trade
SIGNAL_INTERVAL = 2       # generate signal every N candles
MAX_CANDLES = 5000        # max candles to fetch per batch
MAX_BATCHES = 20          # pagination depth (20 x 5000 = 100k candles max)
MIN_CONFIDENCE_OVERRIDE = 0.30  # lower bar during early training
MAX_SIGNALS_PER_HOUR = 50       # scalping signal cap


async def fetch_and_train_instrument(
    client: DerivClient,
    instrument: dict,
    engine: LearningEngine,
    timeframe: str,
    run_id: int,
    verbose: bool = True,
) -> dict:
    """Fetch historical data for one instrument+timeframe and train on it."""
    symbol = instrument["symbol"]
    display_name = instrument["display_name"]
    market = instrument["market"]
    pip_size = PIP_SIZES.get(market, 0.0001)
    granularity = TIMEFRAMES[timeframe]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Run {run_id} | {display_name} ({symbol}) [{timeframe}] [{market}]")
        print(f"{'='*60}")

    # Fetch historical data
    candles = await client.fetch_all_history(
        symbol, granularity=granularity, max_batches=MAX_BATCHES)

    if len(candles) < MIN_HISTORY + MAX_HOLD_BARS:
        if verbose:
            print(f"  Not enough data: {len(candles)} candles (need {MIN_HISTORY + MAX_HOLD_BARS})")
        return {
            "symbol": symbol, "timeframe": timeframe,
            "candles_fetched": len(candles),
            "signals_generated": 0, "wins": 0, "losses": 0,
            "pnl_pips": 0, "status": "insufficient_data",
        }

    if verbose:
        first_ts = datetime.fromtimestamp(candles[0]["epoch"], UTC)
        last_ts = datetime.fromtimestamp(candles[-1]["epoch"], UTC)
        print(f"  Fetched {len(candles)} candles ({first_ts.date()} to {last_ts.date()})")

    # Walk forward through candles
    signals_generated = 0
    wins = 0
    losses = 0
    breakevens = 0
    total_pnl = 0.0
    signal_count_this_hour = 0
    last_hour = -1
    filtered_count = 0

    for i in range(MIN_HISTORY, len(candles) - MAX_HOLD_BARS - 1):
        if i % SIGNAL_INTERVAL != 0:
            continue

        # Cap signals per hour (scalping: ~40-50/hour)
        current_hour = candles[i]["epoch"] // 3600
        if current_hour != last_hour:
            signal_count_this_hour = 0
            last_hour = current_hour
        if signal_count_this_hour >= MAX_SIGNALS_PER_HOUR:
            continue
        signal_count_this_hour += 1

        history = candles[:i + 1]
        signal = generate_signal(history, pip_size=pip_size, min_history=MIN_HISTORY)

        if signal is None or signal.direction == "HOLD":
            continue

        # Apply learned TP/SL adjustments from prior runs
        adj = engine.tp_adjustments.get(signal.regime)
        if adj is not None:
            signal.recommended_tp = max(5, signal.recommended_tp * adj.tp_multiplier / 1.5)
            signal.recommended_sl = max(3, signal.recommended_sl * adj.sl_multiplier / 0.8)

        # After 100+ outcomes, apply learned filters; before that, take low-confidence signals to learn
        if engine.total_signals >= 100:
            should_take, filter_reason = engine.should_take_signal(
                signal.direction, signal.confidence, signal.regime,
                signal.tool_scores)
            if not should_take:
                filtered_count += 1
                continue
        elif signal.confidence < MIN_CONFIDENCE_OVERRIDE:
            filtered_count += 1
            continue

        future = candles[i + 1: i + 1 + MAX_HOLD_BARS]
        outcome = simulate_trade(
            signal.direction, signal.price,
            signal.recommended_tp, signal.recommended_sl,
            future, pip_size=pip_size, max_hold_bars=MAX_HOLD_BARS)

        signals_generated += 1
        total_pnl += outcome.pnl_pips

        if outcome.outcome == "win":
            wins += 1
        elif outcome.outcome == "loss":
            losses += 1
        else:
            breakevens += 1

        # Learn from this outcome — this is the per-signal improvement
        engine.learn_from_failure(
            tool_scores=signal.tool_scores,
            regime=signal.regime,
            failure_category=outcome.failure_category,
            pnl_pips=outcome.pnl_pips,
            epoch=signal.epoch,
            tp_pips=signal.recommended_tp,
            sl_pips=signal.recommended_sl,
        )

        if signals_generated % 100 == 0 and verbose:
            win_rate = wins / max(signals_generated, 1) * 100
            print(f"  [{signals_generated} signals] WR: {win_rate:.1f}% | "
                  f"PnL: {total_pnl:.1f} pips | "
                  f"Patterns: {len(engine.patterns)} | "
                  f"Filtered: {filtered_count}")

    win_rate = wins / max(signals_generated, 1) * 100
    if verbose:
        print(f"\n  RESULTS Run {run_id}: {display_name} [{timeframe}]")
        print(f"    Signals: {signals_generated} (filtered: {filtered_count})")
        print(f"    Wins: {wins} | Losses: {losses} | Breakeven: {breakevens}")
        print(f"    Win Rate: {win_rate:.1f}%")
        print(f"    Total PnL: {total_pnl:.1f} pips")
        print(f"    Patterns Learned (cumulative): {len(engine.patterns)}")
        print(f"    Best Win Streak: {engine.best_win_streak}")
        print(f"    Worst Loss Streak: {engine.worst_loss_streak}")

        if engine.patterns:
            print(f"\n    Top Failure Patterns:")
            sorted_patterns = sorted(
                engine.patterns.values(),
                key=lambda p: -p.occurrence_count)[:5]
            for p in sorted_patterns:
                print(f"      [{p.occurrence_count}x] {p.failure_category} "
                      f"in {p.conditions.get('regime','?')} "
                      f"(avg PnL: {p.avg_pnl_pips:.1f} pips)")
                print(f"        Rule: {p.avoidance_rule[:100]}...")

        print(f"\n    Learned Thresholds by Regime:")
        for regime in ["TRENDING", "RANGING", "VOLATILE", "CHOPPY", "BREAKOUT"]:
            threshold = engine.get_recommended_threshold(regime)
            updates = engine.q_updates.get(regime, 0)
            print(f"      {regime}: {threshold:.2f} ({updates} updates)")

    return {
        "symbol": symbol,
        "display_name": display_name,
        "market": market,
        "timeframe": timeframe,
        "candles_fetched": len(candles),
        "signals_generated": signals_generated,
        "filtered": filtered_count,
        "wins": wins,
        "losses": losses,
        "breakevens": breakevens,
        "win_rate": round(win_rate, 2),
        "pnl_pips": round(total_pnl, 2),
        "patterns_learned": len(engine.patterns),
        "status": "completed",
    }


def _save_training_state(engine: LearningEngine, results: list[dict],
                          run_id: Optional[int] = None):
    """Save the learned state to JSON files."""
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    engine_state = engine.to_dict()
    with open(os.path.join(output_dir, "learned_state.json"), "w") as f:
        json.dump(engine_state, f, indent=2)

    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now(UTC).isoformat(),
            "results": results,
            "stats": engine_state["stats"],
        }, f, indent=2)

    patterns = {}
    for name, p in engine.patterns.items():
        patterns[name] = {
            "pattern_name": p.pattern_name,
            "conditions": p.conditions,
            "failure_category": p.failure_category,
            "occurrence_count": p.occurrence_count,
            "avg_pnl_pips": p.avg_pnl_pips,
            "avoidance_rule": p.avoidance_rule,
            "first_seen": p.first_seen,
            "last_seen": p.last_seen,
        }
    with open(os.path.join(output_dir, "learned_patterns.json"), "w") as f:
        json.dump(patterns, f, indent=2)

    thresholds = {}
    for regime in ["TRENDING", "RANGING", "VOLATILE", "CHOPPY", "BREAKOUT", "UNKNOWN"]:
        thresholds[regime] = {
            "threshold": engine.get_recommended_threshold(regime),
            "updates": engine.q_updates.get(regime, 0),
        }
    with open(os.path.join(output_dir, "learned_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # Per-run results log so you can watch the win rate climb across runs
    if run_id is not None:
        run_file = os.path.join(output_dir, f"run_{run_id}_results.json")
        with open(run_file, "w") as f:
            json.dump({
                "run_id": run_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "results": results,
                "stats": engine_state["stats"],
            }, f, indent=2)


def _load_engine_state() -> Optional[dict]:
    """Load a previously saved LearningEngine state, if present."""
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    state_file = os.path.join(output_dir, "learned_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return None


def audit_training_scope(instruments: list[dict], timeframes: list[str]) -> dict:
    """Build a machine-readable scope audit before any training starts."""
    markets: dict[str, dict] = {}
    for instrument in instruments:
        market = instrument["market"]
        submarket = instrument["submarket"]
        market_bucket = markets.setdefault(market, {"count": 0, "submarkets": {}})
        market_bucket["count"] += 1
        sub_bucket = market_bucket["submarkets"].setdefault(submarket, [])
        sub_bucket.append({
            "symbol": instrument["symbol"],
            "display_name": instrument["display_name"],
        })
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "instrument_count": len(instruments),
        "timeframes": timeframes,
        "timeframe_count": len(timeframes),
        "combo_count": len(instruments) * len(timeframes),
        "markets": markets,
        "deriv_app_id": 1089,
        "max_batches": MAX_BATCHES,
        "max_candles_per_combo": MAX_CANDLES * MAX_BATCHES,
        "min_history": MIN_HISTORY,
        "max_hold_bars": MAX_HOLD_BARS,
        "signal_interval": SIGNAL_INTERVAL,
        "max_signals_per_hour": MAX_SIGNALS_PER_HOUR,
    }


def _write_scope_audit(instruments: list[dict], timeframes: list[str]) -> dict:
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    audit = audit_training_scope(instruments, timeframes)
    with open(os.path.join(output_dir, "training_scope_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    return audit


def _filter_instruments(symbols: Optional[list[str]] = None, markets: Optional[list[str]] = None) -> list[dict]:
    instruments = get_all_instruments()
    if symbols:
        wanted = {s.upper() for s in symbols}
        instruments = [i for i in instruments if i["symbol"].upper() in wanted or i["display_name"].upper() in wanted]
    if markets:
        wanted_markets = {m.lower() for m in markets}
        instruments = [i for i in instruments if i["market"].lower() in wanted_markets or i["submarket"].lower() in wanted_markets]
    return instruments


async def run_single_run(run_id: int, engine: LearningEngine,
                         verbose: bool = True,
                         instruments: Optional[list[dict]] = None,
                         timeframes: Optional[list[str]] = None) -> tuple[LearningEngine, list[dict]]:
    """Run one pass across selected instruments x selected timeframes."""
    instruments = instruments or get_all_instruments()
    timeframes = timeframes or get_all_timeframes()
    total_combos = len(instruments) * len(timeframes)

    print(f"\n{'#'*70}")
    print(f"# RUN {run_id} — {total_combos} instrument-timeframe combos")
    print(f"# Patterns carried in: {len(engine.patterns)} | "
          f"Signals seen: {engine.total_signals}")
    print(f"{'#'*70}")

    all_results = []
    client = DerivClient()
    await client.connect()

    try:
        combo_idx = 0
        for instrument in instruments:
            for tf in timeframes:
                combo_idx += 1
                print(f"\n[{combo_idx}/{total_combos}] ", end="")
                result = await fetch_and_train_instrument(
                    client, instrument, engine, tf, run_id, verbose=verbose)
                all_results.append(result)
                _save_training_state(engine, all_results, run_id=run_id)
    finally:
        await client.close()

    total_signals = sum(r["signals_generated"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    total_pnl = sum(r["pnl_pips"] for r in all_results)
    overall_wr = total_wins / max(total_signals, 1) * 100

    print(f"\n{'='*70}")
    print(f"RUN {run_id} SUMMARY")
    print(f"{'='*70}")
    print(f"  Combos trained: {len(all_results)}")
    print(f"  Total signals:  {total_signals}")
    print(f"  Total wins:     {total_wins}")
    print(f"  Total losses:   {total_losses}")
    print(f"  Overall WR:     {overall_wr:.1f}%")
    print(f"  Total PnL:      {total_pnl:.1f} pips")
    print(f"  Patterns:       {len(engine.patterns)}")
    print(f"  Best win streak:    {engine.best_win_streak}")
    print(f"  Worst loss streak:  {engine.worst_loss_streak}")

    return engine, all_results


async def run_full_training(num_runs: int = 15, verbose: bool = True,
                            symbols: Optional[list[str]] = None,
                            markets: Optional[list[str]] = None,
                            timeframes_filter: Optional[list[str]] = None,
                            resume: bool = True):
    """Run forward-test training with per-run learning carry-over."""
    print("=" * 70)
    print(f"VEILCREAN SIGNAL TRAINING SYSTEM — {num_runs}-RUN FORWARD TEST")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print(f"Using Deriv Public API (app_id 1089)")
    print(f"Timeframes: {', '.join(get_all_timeframes())}")
    print(f"Signal interval: every {SIGNAL_INTERVAL} candles "
          f"(~{MAX_SIGNALS_PER_HOUR} signals/hour cap)")
    print(f"Max hold: {MAX_HOLD_BARS} candles")
    print(f"Max batches per combo: {MAX_BATCHES}")
    print(f"Runs: {num_runs} (each carries forward learned state)")
    print("=" * 70)

    instruments = _filter_instruments(symbols=symbols, markets=markets)
    timeframes = timeframes_filter or get_all_timeframes()
    invalid_tfs = [tf for tf in timeframes if tf not in TIMEFRAMES]
    if invalid_tfs:
        raise ValueError(f"Unsupported timeframes: {invalid_tfs}. Supported: {get_all_timeframes()}")
    if not instruments:
        raise ValueError("No instruments selected for training")
    audit = _write_scope_audit(instruments, timeframes)
    print(f"\nTotal instruments: {len(instruments)}")
    for market in set(i["market"] for i in instruments):
        market_syms = [i for i in instruments if i["market"] == market]
        print(f"  {market}: {len(market_syms)} instruments")
    print(f"Total timeframes: {len(timeframes)}")
    print(f"Total combos: {len(instruments) * len(timeframes)}")

    # Load carried-over engine state from a previous full run, if present
    prior_state = _load_engine_state() if resume else None
    engine = LearningEngine.from_dict(prior_state) if prior_state else LearningEngine()
    if prior_state:
        print(f"\nLoaded prior learned state: {len(engine.patterns)} patterns, "
              f"{engine.total_signals} prior signals")

    run_summaries = []

    for run_id in range(1, num_runs + 1):
        engine, results = await run_single_run(
            run_id, engine, verbose=verbose, instruments=instruments, timeframes=timeframes)

        summary = {
            "run_id": run_id,
            "total_signals": sum(r["signals_generated"] for r in results),
            "total_wins": sum(r["wins"] for r in results),
            "total_losses": sum(r["losses"] for r in results),
            "total_pnl": sum(r["pnl_pips"] for r in results),
            "patterns_learned": len(engine.patterns),
            "best_win_streak": engine.best_win_streak,
            "worst_loss_streak": engine.worst_loss_streak,
        }
        summary["win_rate"] = round(
            summary["total_wins"] / max(summary["total_signals"], 1) * 100, 2)
        run_summaries.append(summary)

        # Save the engine state after each run so the next run carries it forward
        _save_training_state(engine, results, run_id=run_id)

        # Print the running comparison
        print(f"\n{'-'*70}")
        print(f"RUN-BY-RUN PROGRESS (after run {run_id}):")
        print(f"{'Run':<5} {'Signals':>9} {'Wins':>7} {'Losses':>8} "
              f"{'Win%':>7} {'PnL':>11} {'Patterns':>9}")
        print("-" * 60)
        for s in run_summaries:
            print(f"{s['run_id']:<5} {s['total_signals']:>9} {s['total_wins']:>7} "
                  f"{s['total_losses']:>8} {s['win_rate']:>6.1f}% "
                  f"{s['total_pnl']:>10.1f}p {s['patterns_learned']:>9}")
        print(f"{'-'*70}")

    # Final cross-run summary
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_summaries.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now(UTC).isoformat(),
            "runs": run_summaries,
            "final_stats": engine.to_dict()["stats"],
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ALL {num_runs} RUNS COMPLETE — FINAL CROSS-RUN SUMMARY")
    print(f"{'='*70}")
    print(f"{'Run':<5} {'Signals':>9} {'Wins':>7} {'Losses':>8} "
          f"{'Win%':>7} {'PnL':>11} {'Patterns':>9}")
    print("-" * 60)
    for s in run_summaries:
        print(f"{s['run_id']:<5} {s['total_signals']:>9} {s['total_wins']:>7} "
              f"{s['total_losses']:>8} {s['win_rate']:>6.1f}% "
              f"{s['total_pnl']:>10.1f}p {s['patterns_learned']:>9}")

    if len(run_summaries) >= 2:
        first = run_summaries[0]
        last = run_summaries[-1]
        print(f"\nImprovement from Run 1 -> Run {last['run_id']}:")
        print(f"  Win rate: {first['win_rate']:.1f}% -> {last['win_rate']:.1f}%")
        print(f"  Patterns learned: {first['patterns_learned']} -> {last['patterns_learned']}")
        print(f"  PnL: {first['total_pnl']:.1f} -> {last['total_pnl']:.1f} pips")

    print(f"\nAll training data saved to training/output/")
    print(f"  - learned_state.json (full engine memory — load this to resume)")
    print(f"  - learned_patterns.json (failure pattern library)")
    print(f"  - learned_thresholds.json (Q-learned thresholds per regime)")
    print(f"  - run_N_results.json (per-run breakdown)")
    print(f"  - run_summaries.json (cross-run comparison)")

    return engine, run_summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Veilcrean agents on Deriv historical candles.")
    parser.add_argument("--runs", type=int, default=15, help="Number of forward-training passes to run.")
    parser.add_argument("--symbols", nargs="*", help="Optional Deriv symbols/display names to train, e.g. frxEURUSD R_100.")
    parser.add_argument("--markets", nargs="*", help="Optional markets/submarkets to train, e.g. forex volatility.")
    parser.add_argument("--timeframes", nargs="*", help="Optional timeframe labels from the configured list.")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-combo logging.")
    parser.add_argument("--fresh-state", action="store_true", help="Start learning from an empty state instead of resuming output/learned_state.json.")
    parser.add_argument("--max-batches", type=int, help="Override Deriv pagination depth for quick audits/smoke training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.max_batches is not None:
        MAX_BATCHES = args.max_batches
    asyncio.run(run_full_training(
        num_runs=args.runs,
        verbose=not args.quiet,
        symbols=args.symbols,
        markets=args.markets,
        timeframes_filter=args.timeframes,
        resume=not args.fresh_state,
    ))
