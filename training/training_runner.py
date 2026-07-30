"""
Training runner — orchestrates the full forward-test training.

For each instrument:
1. Fetches maximum historical candle data from Deriv (1-minute granularity)
2. Walks forward candle-by-candle (oldest to newest)
3. At each candle, generates a signal using all indicators
4. Simulates the trade on future candles
5. Learns from the outcome (failure analysis + Q-learning threshold update)
6. Applies learned patterns to filter future signals
7. Saves all data to Supabase and JSON files

The agent improves per signal: each failure teaches it what to avoid next time.
"""
from __future__ import annotations

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

from training.deriv_client import DerivClient, get_all_instruments
from training.signal_generator import generate_signal, Signal
from training.trade_simulator import simulate_trade, TradeOutcome
from training.learning_engine import LearningEngine


# Supabase config (read from env)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

# Pip sizes per market
PIP_SIZES = {
    "forex": 0.0001,
    "commodity": 0.01,    # metals
    "cryptocurrency": 1.0, # crypto
    "synthetic_index": 0.001,  # synthetics
}

# Training config
GRANULARITY = 300         # 5-minute candles (faster training, more history)
MIN_HISTORY = 50          # min candles before generating signals
MAX_HOLD_BARS = 20        # max candles to hold a trade
SIGNAL_INTERVAL = 2      # generate signal every N candles (to get ~10-15/hour on 5min)
MAX_CANDLES = 5000       # max candles to fetch per batch
MIN_CONFIDENCE_OVERRIDE = 0.30  # lower bar during training to learn from more signals


async def fetch_and_train_instrument(
    client: DerivClient,
    instrument: dict,
    engine: LearningEngine,
    training_run_id: str,
    verbose: bool = True,
) -> dict:
    """Fetch historical data for one instrument and train on it."""
    symbol = instrument["symbol"]
    display_name = instrument["display_name"]
    market = instrument["market"]
    pip_size = PIP_SIZES.get(market, 0.0001)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {display_name} ({symbol}) [{market}]")
        print(f"{'='*60}")

    # Fetch historical data
    candles = await client.fetch_all_history(
        symbol, granularity=GRANULARITY, max_batches=2)

    if len(candles) < MIN_HISTORY + MAX_HOLD_BARS:
        if verbose:
            print(f"  Not enough data: {len(candles)} candles (need {MIN_HISTORY + MAX_HOLD_BARS})")
        return {
            "symbol": symbol, "candles_fetched": len(candles),
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
        # Generate signal every SIGNAL_INTERVAL candles
        if i % SIGNAL_INTERVAL != 0:
            continue

        # Rate limit signals per hour (~40-50 max)
        current_hour = candles[i]["epoch"] // 3600
        if current_hour != last_hour:
            signal_count_this_hour = 0
            last_hour = current_hour
        if signal_count_this_hour >= 50:
            continue
        signal_count_this_hour += 1

        # Generate signal from candle history up to this point
        history = candles[:i + 1]
        signal = generate_signal(history, pip_size=pip_size, min_history=MIN_HISTORY)

        if signal is None or signal.direction == "HOLD":
            continue

        # During early training, allow lower confidence signals to learn from them
        # After we have 100+ outcomes, start applying learned filters
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

        # Simulate the trade on future candles
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

        # Learn from this outcome
        engine.learn_from_failure(
            tool_scores=signal.tool_scores,
            regime=signal.regime,
            failure_category=outcome.failure_category,
            pnl_pips=outcome.pnl_pips,
            epoch=signal.epoch,
            tp_pips=signal.recommended_tp,
            sl_pips=signal.recommended_sl,
        )

        # Log progress every 100 signals
        if signals_generated % 100 == 0 and verbose:
            win_rate = wins / max(signals_generated, 1) * 100
            print(f"  [{signals_generated} signals] WR: {win_rate:.1f}% | "
                  f"PnL: {total_pnl:.1f} pips | "
                  f"Patterns: {len(engine.patterns)} | "
                  f"Filtered: {filtered_count}")

    win_rate = wins / max(signals_generated, 1) * 100
    if verbose:
        print(f"\n  RESULTS: {display_name}")
        print(f"    Signals: {signals_generated} (filtered: {filtered_count})")
        print(f"    Wins: {wins} | Losses: {losses} | Breakeven: {breakevens}")
        print(f"    Win Rate: {win_rate:.1f}%")
        print(f"    Total PnL: {total_pnl:.1f} pips")
        print(f"    Patterns Learned: {len(engine.patterns)}")
        print(f"    Best Win Streak: {engine.best_win_streak}")
        print(f"    Worst Loss Streak: {engine.worst_loss_streak}")

        # Show top learned patterns
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

        # Show learned thresholds
        print(f"\n    Learned Thresholds by Regime:")
        for regime in ["TRENDING", "RANGING", "VOLATILE", "CHOPPY", "BREAKOUT"]:
            threshold = engine.get_recommended_threshold(regime)
            updates = engine.q_updates.get(regime, 0)
            print(f"      {regime}: {threshold:.2f} ({updates} updates)")

    return {
        "symbol": symbol,
        "display_name": display_name,
        "market": market,
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


async def run_full_training(verbose: bool = True):
    """Run training across all instruments."""
    print("=" * 70)
    print("VEILCREAN SIGNAL TRAINING SYSTEM")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print(f"Using Deriv Public API (app_id 1089)")
    print(f"Granularity: {GRANULARITY}s (1-minute candles)")
    print(f"Signal interval: every {SIGNAL_INTERVAL} candles (~20-40 signals/hour)")
    print(f"Max hold: {MAX_HOLD_BARS} candles")
    print("=" * 70)

    instruments = get_all_instruments()
    print(f"\nTotal instruments: {len(instruments)}")
    for market in set(i["market"] for i in instruments):
        market_syms = [i for i in instruments if i["market"] == market]
        print(f"  {market}: {len(market_syms)} instruments")

    engine = LearningEngine()
    all_results = []

    client = DerivClient()
    await client.connect()

    try:
        for idx, instrument in enumerate(instruments):
            print(f"\n[{idx + 1}/{len(instruments)}] ", end="")
            result = await fetch_and_train_instrument(
                client, instrument, engine,
                training_run_id="run_1",
                verbose=verbose)
            all_results.append(result)

            # Save intermediate state
            _save_training_state(engine, all_results)

    finally:
        await client.close()

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — FINAL SUMMARY")
    print("=" * 70)

    total_signals = sum(r["signals_generated"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    total_pnl = sum(r["pnl_pips"] for r in all_results)

    print(f"\nInstruments trained: {len(all_results)}")
    print(f"Total signals: {total_signals}")
    print(f"Total wins: {total_wins}")
    print(f"Total losses: {total_losses}")
    print(f"Overall win rate: {total_wins / max(total_signals, 1) * 100:.1f}%")
    print(f"Total PnL: {total_pnl:.1f} pips")
    print(f"Patterns learned: {len(engine.patterns)}")

    print(f"\nPer-instrument results:")
    print(f"{'Symbol':<20} {'Signals':>8} {'Wins':>6} {'Losses':>7} "
          f"{'Win%':>6} {'PnL':>10} {'Status':>12}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['symbol']:<20} {r['signals_generated']:>8} {r['wins']:>6} "
              f"{r['losses']:>7} {r['win_rate']:>5.1f}% {r['pnl_pips']:>9.1f}p "
              f"{r['status']:>12}")

    # Save final state
    _save_training_state(engine, all_results)
    print(f"\nAll training data saved.")

    return engine, all_results


def _save_training_state(engine: LearningEngine, results: list[dict]):
    """Save the learned state to JSON files."""
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save learning engine state
    engine_state = engine.to_dict()
    with open(os.path.join(output_dir, "learned_state.json"), "w") as f:
        json.dump(engine_state, f, indent=2)

    # Save results summary
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now(UTC).isoformat(),
            "results": results,
            "stats": engine_state["stats"],
        }, f, indent=2)

    # Save learned patterns separately
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

    # Save learned thresholds
    thresholds = {}
    for regime in ["TRENDING", "RANGING", "VOLATILE", "CHOPPY", "BREAKOUT", "UNKNOWN"]:
        thresholds[regime] = {
            "threshold": engine.get_recommended_threshold(regime),
            "updates": engine.q_updates.get(regime, 0),
        }
    with open(os.path.join(output_dir, "learned_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_full_training(verbose=True))
