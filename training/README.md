# Veilcrean Signal Training System

## Overview

This module trains the Veilcrean trading agent by simulating signals on real historical data from Deriv's public API. The agent learns from each signal — when a signal fails, it analyzes why and generates an avoidance rule so the next signal doesn't make the same mistake.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Deriv API   │────>│ Signal       │────>│ Trade        │────>│ Learning     │
│ (historical │     │ Generator    │     │ Simulator    │     │ Engine       │
│  candles)   │     │ (20+ tools)  │     │ (TP/SL sim)  │     │ (Q-learning) │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      │
                                                                      v
                                                              ┌──────────────┐
                                                              │ Next Signal  │
                                                              │ (filtered +  │
                                                              │  improved)    │
                                                              └──────────────┘
```

1. **Audit** the selected instrument/timeframe scope and save it to `output/training_scope_audit.json`
2. **Fetch** real historical candle data from Deriv for every configured timeframe, bounded to the last 5 years by default or maximum paginated history with `--max-history`
3. **Walk forward** candle-by-candle from oldest to newest
4. **Generate** a signal at each interval using 20+ technical indicators
5. **Simulate** the trade on future candles (TP/SL/max-hold)
6. **Analyze** why the trade succeeded or failed
7. **Learn** an avoidance rule and update adaptive thresholds via Q-learning
8. **Apply** learned patterns to filter future signals and save JSON outputs

## Instruments (44 total)

- **Forex Majors (7)**: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD
- **Forex Minors (7)**: EUR/GBP, EUR/JPY, GBP/JPY, EUR/CHF, AUD/JPY, EUR/AUD, CAD/CHF
- **Metals (4)**: Gold, Silver, Platinum, Palladium
- **Crypto (6)**: BTC, ETH, LTC, BNB, EOS, XRP
- **Volatility Indices (10)**: R_10-R_100, JD_10-JD_100
- **Boom/Crash (4)**: Boom 500/1000, Crash 500/1000
- **Range Break (1)**: Step Range Break
- **Drift Switch (5)**: Drift Switch 10/25/50/75/100

## Running

```bash
pip install numpy pandas websockets
python -m training.training_runner

# Audited smoke run on one real Deriv market/timeframe
python -m training.training_runner --runs 1 --symbols frxEURUSD --timeframes 24h --quiet

# Resume broader scoped training, e.g. all volatility synthetics on 1m and 5m
python -m training.training_runner --runs 15 --markets volatility --timeframes 1m 5m

# Full requested scope in one process: all 44 configured instruments x all 9
# timeframes x 15 runs, using the last 5 years of real Deriv history where
# available. This is very large and saves state after every combo.
python -m training.training_runner --runs 15 --history-years 5

# Recommended operational full run: process the next pending 1-3 instruments
# across all 9 timeframes for 15 runs, then re-run until progress says complete.
python -m training.chunked_full_training --batch-size 3 --runs 15 --history-years 5

# Use maximum Deriv paginated history instead of the 5-year cap.
python -m training.training_runner --runs 15 --max-history
python -m training.chunked_full_training --batch-size 3 --runs 15 --max-history
```

## Training Results (First Run)

| Metric | Value |
|--------|-------|
| Instruments trained | 15 (forex + metals) |
| Total signals | 153 |
| Wins | 52 |
| Losses | 101 |
| Win rate | 34.0% |
| Total PnL | +201 pips |
| Patterns learned | 62 |
| Best win streak | 8 |
| Worst loss streak | 13 |

### Key Learnings

- **RANGING markets**: "wrong_direction" is the most common failure (8x) — the agent learned to avoid signals when indicators conflict in ranging markets
- **TRENDING markets**: SL was too close (3x) — the agent widened SL from 0.8x to 1.58x ATR
- **TRENDING markets**: TP was too far (3x) — the agent reduced TP from 1.5x to 0.95x ATR
- **Adaptive thresholds**: Q-learning adjusted thresholds per regime based on trade outcomes

## Output Files

- `output/training_scope_audit.json` — Selected markets, symbols, timeframes, and combo counts
- `output/learned_state.json` — Full learning engine state (Q-table, patterns, TP/SL adjustments)
- `output/learned_patterns.json` — All learned failure patterns with avoidance rules
- `output/learned_thresholds.json` — Adaptive confidence thresholds per regime
- `output/training_results.json` — Per-instrument training results

## Resumable Full-Training Progress

`python -m training.chunked_full_training --batch-size 3 --runs 15 --history-years 5` trains the next pending instrument batch using real Deriv candles, carries forward the learned state, and records completed symbols plus cumulative stats in `output/full_training_progress.json`. Re-run the same command until `status` becomes `complete`.
