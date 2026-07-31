#!/usr/bin/env python3
"""Veilcrean v2 Full Training - run this locally or on a server.

Usage:
  python scripts/run_v2_train.py

Options (edit below):
  num_runs: number of forward-training passes (3-5 for quick, 10-15 for thorough)
  symbols: None = all 44 instruments, or ['frxEURUSD', 'R_100'] for specific ones
  timeframes_filter: None = all 9 TFs, or ['5m', '15m', '1h'] for core TFs
  history_years: how much history to fetch from Deriv
  resume: True = continue from prior state, False = fresh start
"""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from training.training_runner import run_full_training

async def main():
    await run_full_training(
        num_runs=5,           # 5 forward passes
        verbose=True,
        symbols=None,          # None = all 44 instruments
        timeframes_filter=['1m', '5m', '15m', '1h'],  # Core TFs
        resume=False,         # Fresh v2 start (set True to continue)
        history_years=2.0,    # 2 years of history
    )

if __name__ == '__main__':
    asyncio.run(main())
