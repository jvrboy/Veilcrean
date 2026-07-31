"""
fetch_dsi_history.py
====================
Downloads maximum historical data for DSI 10, 20, 30 from Deriv.
"""
import asyncio
import os
import sys

# Make the repository importable no matter where this script is run from.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pandas as pd
import time
from python_brain.communication.deriv_client import DerivClient
from python_brain.config import DERIV_CFG, HISTORICAL

async def download_all():
    client = DerivClient(app_id=DERIV_CFG.app_id or 1089, api_token=DERIV_CFG.api_token)
    try:
        await client.connect()
    except Exception as e:
        print(
            f"Could not connect to the Deriv WebSocket API: {e}\n"
            "This script needs internet access to wss://ws.derivws.com. "
            "If you are offline, you can still train with the committed "
            "historical data: python scripts/train_single.py 1HZ50V 5 150"
        )
        raise SystemExit(1)
    
    symbols = ["DSI10", "DSI20", "DSI30"]
    # Deriv symbols might have different names, check docs. 
    # Usually: '10' / '20' / '30' or 'DSI_10'
    # We'll try the common ones.
    
    # 1 year ago in epoch
    end_time = int(time.time())
    start_time = end_time - (1 * 365 * 24 * 3600)
    
    for sym in symbols:
        print(f"Downloading history for {sym}...")
        # Start with H1 for long term, then M15
        for gran in [3600, 900]:
            candles = await client.get_historical_data(sym, gran, start_time, end_time)
            if candles:
                df = pd.DataFrame(candles)
                filename = HISTORICAL / f"{sym}_{gran}.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {len(df)} rows to {filename}")
            else:
                print(f"No data for {sym} at granularity {gran}")
                
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(download_all())
