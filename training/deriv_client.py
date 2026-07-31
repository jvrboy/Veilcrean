"""
Deriv public API client — fetches historical candle data for all instruments.

Uses the Deriv WebSocket API (app_id 1089, public, no auth required).
Supports paginated historical data fetching with rate limiting.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import websockets

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
MAX_CANDLES_PER_REQUEST = 5000
RATE_LIMIT_DELAY = 0.5


# All 9 timeframes requested, mapped to Deriv granularity in seconds.
# Deriv supports: 60,120,180,300,600,900,1800,3600,7200,14400,28800,86400
TIMEFRAMES = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "8h": 28800,
    "24h": 86400,
}


INSTRUMENTS = {
    "forex": {
        "majors": [
            ("frxEURUSD", "EUR/USD"),
            ("frxGBPUSD", "GBP/USD"),
            ("frxUSDJPY", "USD/JPY"),
            ("frxUSDCHF", "USD/CHF"),
            ("frxAUDUSD", "AUD/USD"),
            ("frxUSDCAD", "USD/CAD"),
            ("frxNZDUSD", "NZD/USD"),
        ],
        "minors": [
            ("frxEURGBP", "EUR/GBP"),
            ("frxEURJPY", "EUR/JPY"),
            ("frxGBPJPY", "GBP/JPY"),
            ("frxEURCHF", "EUR/CHF"),
            ("frxAUDJPY", "AUD/JPY"),
            ("frxEURAUD", "EUR/AUD"),
            ("frxCADCHF", "CAD/CHF"),
        ],
    },
    "commodity": {
        "metals": [
            ("frxXAUUSD", "Gold/USD"),
            ("frxXAGUSD", "Silver/USD"),
            ("frxXPTUSD", "Platinum/USD"),
            ("frxXPDUSD", "Palladium/USD"),
        ],
    },
    "cryptocurrency": {
        "crypto": [
            ("cryBTCUSD", "BTC/USD"),
            ("cryETHUSD", "ETH/USD"),
            ("cryLTCUSD", "LTC/USD"),
            ("cryBNBUSD", "BNB/USD"),
            ("cryEOSUSD", "EOS/USD"),
            ("cryXRPUSD", "XRP/USD"),
        ],
    },
    "synthetic_index": {
        "volatility": [
            ("R_10", "Volatility 10"),
            ("R_25", "Volatility 25"),
            ("R_50", "Volatility 50"),
            ("R_75", "Volatility 75"),
            ("R_100", "Volatility 100"),
            ("JD10", "Jump 10"),
            ("JD25", "Jump 25"),
            ("JD50", "Jump 50"),
            ("JD75", "Jump 75"),
            ("JD100", "Jump 100"),
        ],
        "boom_crash": [
            ("BOOM500", "Boom 500"),
            ("BOOM1000", "Boom 1000"),
            ("CRASH500", "Crash 500"),
            ("CRASH1000", "Crash 1000"),
        ],
        "range_break": [
            ("stpRNG", "Step Range Break"),
        ],
        "drift_switch": [
            ("DRSI10", "Drift Switch 10"),
            ("DRSI25", "Drift Switch 25"),
            ("DRSI50", "Drift Switch 50"),
            ("DRSI75", "Drift Switch 75"),
            ("DRSI100", "Drift Switch 100"),
        ],
    },
}


def get_all_instruments() -> list[dict]:
    """Return every instrument across all markets and submarkets."""
    result = []
    for market, submarkets in INSTRUMENTS.items():
        for submarket, symbols in submarkets.items():
            for symbol, display_name in symbols:
                result.append({
                    "symbol": symbol,
                    "display_name": display_name,
                    "market": market,
                    "submarket": submarket,
                })
    return result


def get_all_timeframes() -> list[str]:
    """Return all configured timeframe labels."""
    return list(TIMEFRAMES.keys())


class DerivClient:
    """Async WebSocket client for Deriv's public API."""

    def __init__(self):
        self._ws = None
        self._request_id = 0

    async def connect(self):
        self._ws = await websockets.connect(DERIV_WS_URL, close_timeout=10)
        return self

    async def close(self):
        if self._ws:
            await self._ws.close()

    async def _reconnect(self):
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass
        self._ws = await websockets.connect(DERIV_WS_URL, close_timeout=10)

    async def _send(self, payload: dict) -> dict:
        self._request_id += 1
        payload["req_id"] = self._request_id
        for attempt in range(3):
            try:
                if self._ws is None or getattr(self._ws, 'closed', False) or self._ws.protocol.state.name != 'OPEN':
                    await self._reconnect()
                await self._ws.send(json.dumps(payload))
                while True:
                    resp = json.loads(await self._ws.recv())
                    if resp.get("req_id") == self._request_id:
                        return resp
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1)
                    await self._reconnect()
                else:
                    raise

    async def fetch_candles(self, symbol: str, granularity: int = 60,
                            count: int = MAX_CANDLES_PER_REQUEST,
                            end: Optional[int] = None) -> list[dict]:
        if end is None:
            end = "latest"
        payload = {
            "ticks_history": symbol, "end": end,
            "count": min(count, MAX_CANDLES_PER_REQUEST),
            "style": "candles", "granularity": granularity,
        }
        resp = await self._send(payload)
        if resp.get("error"):
            return []
        return resp.get("candles", [])

    async def fetch_all_history(self, symbol: str, granularity: int = 60,
                                 max_batches: int = 20,
                                 start_epoch: Optional[int] = None) -> list[dict]:
        """Fetch Deriv candle history, optionally bounded by a start epoch.

        max_batches=20 x 5000 candles = up to 100k candles per symbol/timeframe.
        For 1m candles that's ~69 days; for 24h candles that's ~274 years.
        When ``start_epoch`` is supplied, pagination stops once candles reach
        that boundary and any older candles are discarded.
        """
        all_candles: list[dict] = []
        end = "latest"
        for batch_num in range(max_batches):
            candles = await self.fetch_candles(symbol, granularity=granularity, end=end)
            if not candles:
                break
            seen_epochs = {c["epoch"] for c in all_candles}
            new_candles = [c for c in candles if c["epoch"] not in seen_epochs]
            if not new_candles:
                break
            if start_epoch is not None:
                new_candles = [c for c in new_candles if c["epoch"] >= start_epoch]
            all_candles.extend(new_candles)
            oldest = min(c["epoch"] for c in candles)
            if start_epoch is not None and oldest <= start_epoch:
                break
            if len(candles) < 100:
                break
            end = oldest - 1
            await asyncio.sleep(RATE_LIMIT_DELAY)
        all_candles.sort(key=lambda c: c["epoch"])
        return all_candles
