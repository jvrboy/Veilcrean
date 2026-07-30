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

1. **Fetch** historical candle data from Deriv (1-minute or 5-minute granularity)
2. **Walk forward** candle-by-candle from oldest to newest
3. **Generate** a signal at each interval using 20+ technical indicators
4. **Simulate** the trade on future candles (TP/SL/max-hold)
5. **Analyze** why the trade succeeded or failed
6. **Learn** an avoidance rule and update adaptive thresholds via Q-learning
7. **Apply** learned patterns to filter future signals
8. **Save** all data to Supabase and JSON files

## Instruments (40 total)

- **Forex Majors (7)**: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD
- **Forex Minors (7)**: EUR/GBP, EUR/JPY, GBP/JPY, EUR/CHF, AUD/JPY, EUR/AUD, CAD/CHF
- **Metals (4)**: Gold, Silver, Platinum, Palladium
- **Crypto (6)**: BTC, ETH, LTC, BNB, EOS, XRP
- **Volatility Indices (10)**: R_10-R_100, JD_10-JD_100
- **Boom/Crash (4)**: Boom 500/1000, Crash 500/1000
- **Range Break (1)**: Step Range Break

## Running

```bash
pip install numpy pandas websockets
python -m training.training_runner
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

- `output/learned_state.json` — Full learning engine state (Q-table, patterns, TP/SL adjustments)
- `output/learned_patterns.json` — All learned failure patterns with avoidance rules
- `output/learned_thresholds.json` — Adaptive confidence thresholds per regime
- `output/training_results.json` — Per-instrument training results
