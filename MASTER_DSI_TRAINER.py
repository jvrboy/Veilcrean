"""
MASTER_DSI_TRAINER.py
=====================
The ultimate script for mastering Deriv Drift Switch Indices.
Run this on your local machine to fetch real data and train the bot.

Prerequisites:
    1. DERIV_API_TOKEN set in .env
    2. Internet access
"""
import os
import asyncio
import subprocess
import sys
from pathlib import Path

def run_step(name, cmd):
    print(f"\n{'='*20} STEP: {name} {'='*20}")
    try:
        # We use the current python executable
        result = subprocess.run([sys.executable] + cmd, check=True)
        print(f"SUCCESS: {name}")
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {name} with error {e}")
        return False
    return True

async def master_flow():
    # 1. Fetch Real History
    if not run_step("FETCH REAL DSI HISTORY", ["scripts/fetch_dsi_history.py"]):
        print("Aborting. Could not fetch real data from Deriv.")
        return

    # 2. Iterative Training & Backtesting
    # We run 5 generations of training/backtesting
    for gen in range(5):
        print(f"\n\nGENERATION {gen+1}")
        
        # Train
        if not run_step(f"TRAIN GEN {gen+1}", ["scripts/train_dsi_offline.py"]):
            break
            
        # Backtest
        if not run_step(f"BACKTEST GEN {gen+1}", ["scripts/backtest_dsi.py"]):
            break
            
    print("\n\nMASTER TRAINING COMPLETE.")
    print("Check 'models/' for the most profitable v_*_offline_dsi_master model.")

if __name__ == "__main__":
    asyncio.run(master_flow())
