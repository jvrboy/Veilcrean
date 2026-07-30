"""
MASTER_FULL_TRAINER.py
======================
The ultimate script for full forward-test training across ALL instruments
and ALL 9 timeframes, for 15 runs with per-signal learning carry-over.

Covers:
  FOREX:        Majors, Minors
  Metals:       Gold, Silver, Platinum, Palladium
  Crypto:       BTC, ETH, LTC, BNB, EOS, XRP
  Synthetics:   Volatility (R_10..100, Jump 10..100),
                Boom/Crash (500/1000),
                Range Break (Step Range),
                Drift Switch (DSI 10/25/50/75/100)

Timeframes: 1m, 2m, 5m, 15m, 30m, 1h, 4h, 8h, 24h

Per run:
  - Walks forward candle-by-candle over real Deriv historical data
  - Generates scalping signals (~40-50/hour cap)
  - Simulates each trade, learns WHY it failed, and fixes it
  - Carries the full learned state (Q-table, patterns, TP/SL) into the next run
  - Logs wins/losses per run so you can watch the win rate climb

Prerequisites:
    1. pip install -r requirements.txt   (websockets, numpy, pandas)
    2. Internet access to wss://ws.derivws.com
    3. (Optional) SUPABASE_URL + SUPABASE_ANON_KEY in .env for cloud backup

Run:
    python MASTER_FULL_TRAINER.py
"""
import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.deriv_client import get_all_instruments, get_all_timeframes
from training.training_runner import run_full_training


def print_plan():
    instruments = get_all_instruments()
    timeframes = get_all_timeframes()
    print("=" * 70)
    print("VEILCREAN MASTER FULL TRAINER")
    print("=" * 70)
    print(f"Instruments : {len(instruments)}")
    print(f"Timeframes  : {len(timeframes)} ({', '.join(timeframes)})")
    print(f"Combos      : {len(instruments) * len(timeframes)}")
    print(f"Runs        : 15 (each carries forward learned state)")
    print()
    print("Instrument coverage:")
    by_market = {}
    for i in instruments:
        by_market.setdefault(i["market"], []).append(i)
    for market, syms in by_market.items():
        submarkets = {}
        for s in syms:
            submarkets.setdefault(s["submarket"], []).append(s["display_name"])
        print(f"  {market}:")
        for sub, names in submarkets.items():
            print(f"    {sub}: {', '.join(names)}")
    print()
    print("Data source : Deriv public WebSocket API (app_id 1089, no token needed)")
    print("Output      : training/output/ (JSON files + engine memory)")
    print("=" * 70)


def main():
    print_plan()
    asyncio.run(run_full_training(num_runs=15, verbose=True))


if __name__ == "__main__":
    main()
