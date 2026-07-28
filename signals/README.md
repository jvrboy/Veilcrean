# Veilcrean — Self-Learning Signals

A closed-loop signal engine built on Veilcrean's 144-tool `ConfluenceEngine`.
It analyses a **fixed universe of 24 instruments**, emits `ENTRY / TP / SL`
signals, records every signal, scores how past signals performed against real
price action, and adapts its per-instrument parameters so it improves over time.

> ⚠️ Educational / research output — **not financial advice**. Synthetic-index
> signals for Boom/Crash are direction-forced by configuration, not prediction.

## Tracked universe (fixed)

| Group | Instruments | Data source |
|---|---|---|
| FX | EURUSD, GBPUSD, USDCAD | yfinance |
| Metals | XAUUSD, XAGUSD | yfinance |
| Crypto | BTCUSD | yfinance |
| Volatility Index | 10, 15, 50, 75, 100 | Deriv WebSocket (real) |
| Drift Switch Index | 10, 20, 30 | Deriv WebSocket (real) |
| Boom Index | 100, 200, 500, 900, 1000 | Deriv WebSocket (real) — **always BUY** |
| Crash Index | 100, 200, 500, 900, 1000 | Deriv WebSocket (real) — **always SELL** |

## Run

```bash
python -m signals.generate_signals
```

Outputs (all committed to the repo — this is the system's memory):

| File | Purpose |
|---|---|
| `latest_signals.json` / `latest_signals.md` | most recent run |
| `ledger.json` | every signal ever issued + its scored outcome |
| `learning_state.json` | per-instrument adaptive parameters (the learning) |
| `performance_report.md` | win rates + learned parameters per instrument |

## How the self-learning works

Each run:
1. **Score history** — replays real candles since each OPEN signal was issued and
   marks it `WIN`/`LOSS`/`EXPIRED` based on whether TP or SL was touched first.
2. **Adapt** (`learning.py`) — once an instrument has ≥4 graded signals:
   - win rate `< 40%` → raise `threshold`, cut `conf_multiplier`, raise `rr`
     (be more selective);
   - win rate `> 60%` → lower `threshold`, raise `conf_multiplier`
     (be more aggressive);
   - repeated SL hits → widen `k_sl` (more room); repeated TP hits → tighten it.
   All parameters are bounded so learning can't drift into nonsense.
3. **Generate** — fetches real data, runs all 144 tools, and applies the
   instrument's *learned* parameters to produce the signal.

Direction logic: aggregate tool score → BUY/SELL; **Boom forced BUY, Crash
forced SELL**. TP/SL are ATR-based (`SL = k_sl·ATR`, `TP = rr·SL`).

## Dependencies

`yfinance`, `websocket-client`, `pandas`, `numpy` (plus the repo's
`python_brain`). See `signals/requirements.txt`.
