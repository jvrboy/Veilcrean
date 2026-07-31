"""
Training runner v2 - per-instrument, per-timeframe training with R:R optimization.

Improvements over v1:
1. Per-instrument + per-timeframe TP/SL from learning engine
2. Trailing stop and breakeven support
3. Session-aware signal filtering
4. Better progress reporting with expectancy tracking
5. Asymmetric R:R by regime
6. Cooldown logic from learning engine
7. More aggressive exploration in early runs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, UTC, timedelta
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.deriv_client import DerivClient, get_all_instruments, get_all_timeframes, TIMEFRAMES
from training.signal_generator import generate_signal, Signal
from training.trade_simulator import simulate_trade, TradeOutcome
from training.learning_engine import LearningEngine


SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

PIP_SIZES = {
    "forex": 0.0001,
    "commodity": 0.01,
    "cryptocurrency": 1.0,
    "synthetic_index": 0.001,
}

MIN_HISTORY = 50
MAX_HOLD_BARS = 20
SIGNAL_INTERVAL = 2
MAX_CANDLES = 5000
MAX_BATCHES = 20
MIN_CONFIDENCE_OVERRIDE = 0.30
MAX_SIGNALS_PER_HOUR = 50


async def fetch_and_train_instrument(
    client: DerivClient,
    instrument: dict,
    engine: LearningEngine,
    timeframe: str,
    run_id: int,
    verbose: bool = True,
    start_epoch: Optional[int] = None,
) -> dict:
    symbol = instrument["symbol"]
    display_name = instrument["display_name"]
    market = instrument["market"]
    pip_size = PIP_SIZES.get(market, 0.0001)
    granularity = TIMEFRAMES[timeframe]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Run {run_id} | {display_name} ({symbol}) [{timeframe}] [{market}]")
        print(f"{'='*60}")

    candles = await client.fetch_all_history(
        symbol, granularity=granularity, max_batches=MAX_BATCHES,
        start_epoch=start_epoch)

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

    signals_generated = 0
    wins = 0
    losses = 0
    breakevens = 0
    total_pnl = 0.0
    total_tp_pips = 0.0
    total_sl_pips = 0.0
    signal_count_this_hour = 0
    last_hour = -1
    filtered_count = 0

    for i in range(MIN_HISTORY, len(candles) - MAX_HOLD_BARS - 1):
        if i % SIGNAL_INTERVAL != 0:
            continue

        current_hour = candles[i]["epoch"] // 3600
        if current_hour != last_hour:
            signal_count_this_hour = 0
            last_hour = current_hour
        if signal_count_this_hour >= MAX_SIGNALS_PER_HOUR:
            continue
        signal_count_this_hour += 1

        history = candles[:i + 1]

        # Get optimized TP/SL from learning engine
        regime_est = "UNKNOWN"
        if len(history) >= 50:
            last_80 = history[-80:]
            closes_tmp = np.array([c["close"] for c in last_80], dtype=float)
            highs_tmp = np.array([c["high"] for c in last_80], dtype=float)
            lows_tmp = np.array([c["low"] for c in last_80], dtype=float)
            atr_tmp = 0.0
            if len(closes_tmp) >= 15:
                for j in range(-14, 0):
                    atr_tmp = max(atr_tmp, highs_tmp[j] - lows_tmp[j])
                atr_tmp /= 14
            atr_pips_tmp = atr_tmp / pip_size if pip_size > 0 else atr_tmp
            tp_opt, sl_opt, trail_en, be_trigger = engine.get_optimal_tp_sl(
                regime_est, atr_pips_tmp, symbol, timeframe)
        else:
            tp_opt = 0
            sl_opt = 0
            trail_en = False
            be_trigger = 0.7

        signal = generate_signal(
            history, pip_size=pip_size, min_history=MIN_HISTORY,
            tp_override=tp_opt, sl_override=sl_opt,
            trailing_enabled=trail_en, breakeven_trigger=be_trigger)

        if signal is None or signal.direction == "HOLD":
            continue

        regime_est = signal.regime

        # Apply learned filters
        if engine.total_signals >= 100:
            should_take, filter_reason = engine.should_take_signal(
                signal.direction, signal.confidence, signal.regime,
                signal.tool_scores,
                instrument=symbol, timeframe=timeframe,
                epoch=signal.epoch)
            if not should_take and not filter_reason.startswith("Confidence"):
                filtered_count += 1
                continue
            if not should_take and signal.confidence < MIN_CONFIDENCE_OVERRIDE:
                filtered_count += 1
                continue
        elif signal.confidence < MIN_CONFIDENCE_OVERRIDE:
            filtered_count += 1
            continue

        future = candles[i + 1: i + 1 + MAX_HOLD_BARS]

        # Get trailing config from learning engine
        atr_pips = signal.recommended_sl / 0.8 if signal.recommended_sl > 0 else 10
        _, _, trailing_en, be_trigger = engine.get_optimal_tp_sl(
            signal.regime, atr_pips, symbol, timeframe)

        outcome = simulate_trade(
            signal.direction, signal.price,
            signal.recommended_tp, signal.recommended_sl,
            future, pip_size=pip_size, max_hold_bars=MAX_HOLD_BARS,
            trailing_enabled=trailing_en,
            trailing_distance_pct=0.5,
            breakeven_trigger_pct=be_trigger)

        signals_generated += 1
        total_pnl += outcome.pnl_pips

        if outcome.outcome == "win":
            wins += 1
            total_tp_pips += outcome.pnl_pips
        elif outcome.outcome == "loss":
            losses += 1
            total_sl_pips += abs(outcome.pnl_pips)
        else:
            breakevens += 1

        engine.learn_from_failure(
            tool_scores=signal.tool_scores,
            regime=signal.regime,
            failure_category=outcome.failure_category or "breakeven",
            pnl_pips=outcome.pnl_pips,
            epoch=signal.epoch,
            tp_pips=signal.recommended_tp,
            sl_pips=signal.recommended_sl,
            instrument=symbol,
            timeframe=timeframe,
            direction=signal.direction,
        )

        if signals_generated % 100 == 0 and verbose:
            wr = wins / max(signals_generated, 1) * 100
            avg_win = total_tp_pips / max(wins, 1)
            avg_loss = total_sl_pips / max(losses, 1)
            expectancy = (wins/max(signals_generated,1))*avg_win - (losses/max(signals_generated,1))*avg_loss
            pf = total_tp_pips / max(total_sl_pips, 1e-10)
            print(f"  [{signals_generated}] WR:{wr:.1f}% PnL:{total_pnl:.1f}p "
                  f"Exp:{expectancy:.1f}p PF:{pf:.2f} Patterns:{len(engine.patterns)} "
                  f"Filtered:{filtered_count}")

    win_rate = wins / max(signals_generated, 1) * 100
    avg_win = total_tp_pips / max(wins, 1)
    avg_loss = total_sl_pips / max(losses, 1)
    expectancy = (wins/max(signals_generated,1))*avg_win - (losses/max(signals_generated,1))*avg_loss
    profit_factor = total_tp_pips / max(total_sl_pips, 1e-10)

    if verbose:
        print(f"\n  RESULTS Run {run_id}: {display_name} [{timeframe}]")
        print(f"    Signals: {signals_generated} (filtered: {filtered_count})")
        print(f"    Wins: {wins} | Losses: {losses} | Breakeven: {breakevens}")
        print(f"    Win Rate: {win_rate:.1f}%")
        print(f"    Avg Win: {avg_win:.1f}p | Avg Loss: {avg_loss:.1f}p")
        print(f"    Expectancy: {expectancy:.1f}p per signal")
        print(f"    Profit Factor: {profit_factor:.2f}")
        print(f"    Total PnL: {total_pnl:.1f} pips")
        print(f"    Patterns Learned (cumulative): {len(engine.patterns)}")

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
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "profit_factor": round(profit_factor, 2),
        "pnl_pips": round(total_pnl, 2),
        "patterns_learned": len(engine.patterns),
        "status": "completed",
    }


def _save_training_state(engine: LearningEngine, results: list[dict],
                          run_id: Optional[int] = None):
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
            "instrument": p.instrument,
            "timeframe": p.timeframe,
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

    # TP/SL adjustments
    tp_adj = {}
    for key, adj in engine.tp_adjustments.items():
        tp_adj[key] = {
            "regime": adj.regime, "instrument": adj.instrument,
            "timeframe": adj.timeframe,
            "tp_mult": adj.tp_mult, "sl_mult": adj.sl_mult,
            "trailing_enabled": adj.trailing_enabled,
            "win_count": adj.win_count, "loss_count": adj.loss_count,
            "expectancy": adj.expectancy,
        }
    with open(os.path.join(output_dir, "learned_tp_sl.json"), "w") as f:
        json.dump(tp_adj, f, indent=2)

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
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    state_file = os.path.join(output_dir, "learned_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return None


def audit_training_scope(instruments: list[dict], timeframes: list[str]) -> dict:
    markets: dict[str, dict] = {}
    for instrument in instruments:
        market = instrument["market"]
        submarket = instrument["submarket"]
        market_bucket = markets.setdefault(market, {"count": 0, "submarkets": {}})
        market_bucket["count"] += 1
        sub_bucket = market_bucket["submarkets"].setdefault(submarket, [])
        sub_bucket.append({"symbol": instrument["symbol"], "display_name": instrument["display_name"]})
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
        "version": "v2",
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
                         timeframes: Optional[list[str]] = None,
                         start_epoch: Optional[int] = None) -> tuple[LearningEngine, list[dict]]:
    instruments = instruments or get_all_instruments()
    timeframes = timeframes or get_all_timeframes()
    total_combos = len(instruments) * len(timeframes)

    print(f"\n{'#'*70}")
    print(f"# RUN {run_id} — {total_combos} instrument-timeframe combos (v2)")
    print(f"# Patterns: {len(engine.patterns)} | Signals: {engine.total_signals} | Epsilon: {engine.q_epsilon:.4f}")
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
                    client, instrument, engine, tf, run_id, verbose=verbose,
                    start_epoch=start_epoch)
                all_results.append(result)
                _save_training_state(engine, all_results, run_id=run_id)
    finally:
        await client.close()

    total_signals = sum(r["signals_generated"] for r in all_results)
    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    total_tp = sum(r.get("avg_win", 0) * r["wins"] for r in all_results)
    total_sl = sum(r.get("avg_loss", 0) * r["losses"] for r in all_results)
    total_pnl = sum(r["pnl_pips"] for r in all_results)
    overall_wr = total_wins / max(total_signals, 1) * 100
    pf = total_tp / max(total_sl, 1e-10)
    avg_w = total_tp / max(total_wins, 1)
    avg_l = total_sl / max(total_losses, 1)
    exp = (total_wins/max(total_signals,1))*avg_w - (total_losses/max(total_signals,1))*avg_l

    print(f"\n{'='*70}")
    print(f"RUN {run_id} SUMMARY")
    print(f"{'='*70}")
    print(f"  Combos trained: {len(all_results)}")
    print(f"  Total signals:  {total_signals}")
    print(f"  Total wins:     {total_wins}")
    print(f"  Total losses:   {total_losses}")
    print(f"  Win Rate:       {overall_wr:.1f}%")
    print(f"  Avg Win:        {avg_w:.1f} pips")
    print(f"  Avg Loss:       {avg_l:.1f} pips")
    print(f"  Expectancy:     {exp:.2f} pips/signal")
    print(f"  Profit Factor:  {pf:.2f}")
    print(f"  Total PnL:      {total_pnl:.1f} pips")
    print(f"  Patterns:       {len(engine.patterns)}")
    print(f"  Best win streak:    {engine.best_win_streak}")
    print(f"  Worst loss streak:  {engine.worst_loss_streak}")

    return engine, all_results


async def run_full_training(num_runs: int = 15, verbose: bool = True,
                            symbols: Optional[list[str]] = None,
                            markets: Optional[list[str]] = None,
                            timeframes_filter: Optional[list[str]] = None,
                            resume: bool = True,
                            history_years: Optional[float] = 5.0):
    print("=" * 70)
    print(f"VEILCREAN SIGNAL TRAINING SYSTEM v2 — {num_runs}-RUN FORWARD TEST")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print(f"Using Deriv Public API (app_id 1089)")
    print(f"Features: 35 indicators, per-instrument learning, trailing stops")
    print(f"Timeframes: {', '.join(get_all_timeframes())}")
    print(f"Signal interval: every {SIGNAL_INTERVAL} candles")
    print(f"Max hold: {MAX_HOLD_BARS} candles")
    if history_years is None:
        start_epoch = None
        print("History: maximum available from Deriv")
    else:
        start_epoch = int((datetime.now(UTC) - timedelta(days=365.25 * history_years)).timestamp())
        print(f"History: last {history_years:g} years")
    print(f"Runs: {num_runs} (each carries forward learned state)")
    print("=" * 70)

    instruments = _filter_instruments(symbols=symbols, markets=markets)
    timeframes = timeframes_filter or get_all_timeframes()
    invalid_tfs = [tf for tf in timeframes if tf not in TIMEFRAMES]
    if invalid_tfs:
        raise ValueError(f"Unsupported timeframes: {invalid_tfs}")
    if not instruments:
        raise ValueError("No instruments selected")
    audit = _write_scope_audit(instruments, timeframes)
    audit["history_years"] = history_years
    audit["start_epoch"] = start_epoch
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    with open(os.path.join(output_dir, "training_scope_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nTotal instruments: {len(instruments)}")
    for market in set(i["market"] for i in instruments):
        market_syms = [i for i in instruments if i["market"] == market]
        print(f"  {market}: {len(market_syms)}")
    print(f"Total timeframes: {len(timeframes)}")
    print(f"Total combos: {len(instruments) * len(timeframes)}")

    prior_state = _load_engine_state() if resume else None
    engine = LearningEngine.from_dict(prior_state) if prior_state else LearningEngine()
    if prior_state:
        print(f"\nLoaded prior state: {len(engine.patterns)} patterns, {engine.total_signals} signals")
        stats = prior_state.get("stats", {})
        print(f"  Prior WR: {stats.get('win_rate', 0):.1%} | Prior PnL: {stats.get('total_pnl', 0):.1f}p")

    run_summaries = []

    for run_id in range(1, num_runs + 1):
        engine, results = await run_single_run(
            run_id, engine, verbose=verbose, instruments=instruments,
            timeframes=timeframes, start_epoch=start_epoch)

        total_signals = sum(r["signals_generated"] for r in results)
        total_wins = sum(r["wins"] for r in results)
        total_losses = sum(r["losses"] for r in results)
        total_pnl = sum(r["pnl_pips"] for r in results)
        total_tp = sum(r.get("avg_win", 0) * r["wins"] for r in all_results if r.get("wins", 0) > 0) if 'all_results' in dir() else sum(r.get("avg_win", 0) * r["wins"] for r in results if r.get("wins", 0) > 0)
        total_sl = sum(r.get("avg_loss", 0) * r["losses"] for r in results if r.get("losses", 0) > 0)
        avg_w = total_tp / max(total_wins, 1)
        avg_l = total_sl / max(total_losses, 1)
        exp_val = (total_wins/max(total_signals,1))*avg_w - (total_losses/max(total_signals,1))*avg_l
        pf = total_tp / max(total_sl, 1e-10)

        summary = {
            "run_id": run_id,
            "total_signals": total_signals,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_pnl": total_pnl,
            "avg_win_pips": round(avg_w, 2),
            "avg_loss_pips": round(avg_l, 2),
            "expectancy": round(exp_val, 2),
            "profit_factor": round(pf, 2),
            "patterns_learned": len(engine.patterns),
            "best_win_streak": engine.best_win_streak,
            "worst_loss_streak": engine.worst_loss_streak,
        }
        summary["win_rate"] = round(total_wins / max(total_signals, 1) * 100, 2)
        run_summaries.append(summary)

        _save_training_state(engine, results, run_id=run_id)

        print(f"\n{'-'*70}")
        print(f"RUN-BY-RUN PROGRESS:")
        print(f"{'Run':<5} {'Signals':>9} {'WR%':>7} {'AvgW':>7} {'AvgL':>7} {'Exp':>7} {'PF':>7} {'PnL':>11}")
        print("-" * 65)
        for s in run_summaries:
            print(f"{s['run_id']:<5} {s['total_signals']:>9} {s['win_rate']:>6.1f}% "
                  f"{s.get('avg_win_pips',0):>6.1f}p {s.get('avg_loss_pips',0):>6.1f}p "
                  f"{s.get('expectancy',0):>6.1f}p {s.get('profit_factor',0):>6.2f} "
                  f"{s['total_pnl']:>10.1f}p")
        print(f"{'-'*70}")

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_summaries.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now(UTC).isoformat(),
            "runs": run_summaries,
            "final_stats": engine.to_dict()["stats"],
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ALL {num_runs} RUNS COMPLETE — FINAL SUMMARY")
    print(f"{'='*70}")
    for s in run_summaries:
        print(f"  Run {s['run_id']}: WR={s['win_rate']:.1f}% Exp={s.get('expectancy',0):.1f}p PF={s.get('profit_factor',0):.2f} PnL={s['total_pnl']:.1f}p")

    if len(run_summaries) >= 2:
        first = run_summaries[0]
        last = run_summaries[-1]
        print(f"\nImprovement Run 1 -> Run {last['run_id']}:")
        print(f"  Win rate: {first['win_rate']:.1f}% -> {last['win_rate']:.1f}%")
        print(f"  Expectancy: {first.get('expectancy',0):.1f}p -> {last.get('expectancy',0):.1f}p")
        print(f"  PnL: {first['total_pnl']:.1f} -> {last['total_pnl']:.1f} pips")

    print(f"\nAll data saved to training/output/")
    return engine, run_summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Veilcrean v2 agents on Deriv historical candles.")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--markets", nargs="*")
    parser.add_argument("--timeframes", nargs="*")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fresh-state", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--history-years", type=float, default=5.0)
    parser.add_argument("--max-history", action="store_true")
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
        history_years=None if args.max_history else args.history_years,
    ))
