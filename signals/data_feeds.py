"""
data_feeds.py
=============
Fetches REAL market data for every instrument the Veilcrean signal engine
covers and returns multi-timeframe OHLCV buffers ready for the analysis tools.

Two live sources are used (no simulated data):
  * yfinance          -> FX, metals, crypto (EURUSD, GBPUSD, USDCAD, XAUUSD,
                         XAGUSD, BTCUSD)
  * Deriv WebSocket   -> synthetic indices (Volatility, Drift Switch,
                         Boom & Crash). Public app_id, no auth required.
"""
from __future__ import annotations

import json
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Instrument registry
# --------------------------------------------------------------------------- #
# kind: "yf" (yfinance) or "deriv" (Deriv WebSocket)
# force: optional forced direction ("BUY"/"SELL") -> Boom always buy, Crash sell
INSTRUMENTS: Dict[str, dict] = {
    # ---- FX / metals / crypto (yfinance) --------------------------------- #
    "EURUSD":  {"kind": "yf",    "ticker": "EURUSD=X"},
    "GBPUSD":  {"kind": "yf",    "ticker": "GBPUSD=X"},
    "USDCAD":  {"kind": "yf",    "ticker": "USDCAD=X"},
    "XAUUSD":  {"kind": "yf",    "ticker": "GC=F"},
    "XAGUSD":  {"kind": "yf",    "ticker": "SI=F"},
    "BTCUSD":  {"kind": "yf",    "ticker": "BTC-USD"},
    # ---- Volatility indices (Deriv) -------------------------------------- #
    "Volatility 10 Index":  {"kind": "deriv", "symbol": "R_10"},
    "Volatility 15 Index":  {"kind": "deriv", "symbol": "R_15"},
    "Volatility 50 Index":  {"kind": "deriv", "symbol": "R_50"},
    "Volatility 75 Index":  {"kind": "deriv", "symbol": "R_75"},
    "Volatility 100 Index": {"kind": "deriv", "symbol": "R_100"},
    # ---- Drift Switch indices (Deriv) ------------------------------------ #
    "Drift Switch 10 Index": {"kind": "deriv", "symbol": "DSI10"},
    "Drift Switch 20 Index": {"kind": "deriv", "symbol": "DSI20"},
    "Drift Switch 30 Index": {"kind": "deriv", "symbol": "DSI30"},
    # ---- Boom indices (Deriv) -> ALWAYS BUY ------------------------------ #
    "Boom 100 Index":  {"kind": "deriv", "symbol": "BOOM100",  "force": "BUY"},
    "Boom 200 Index":  {"kind": "deriv", "symbol": "BOOM200",  "force": "BUY"},
    "Boom 500 Index":  {"kind": "deriv", "symbol": "BOOM500",  "force": "BUY"},
    "Boom 900 Index":  {"kind": "deriv", "symbol": "BOOM900",  "force": "BUY"},
    "Boom 1000 Index": {"kind": "deriv", "symbol": "BOOM1000", "force": "BUY"},
    # ---- Crash indices (Deriv) -> ALWAYS SELL ---------------------------- #
    "Crash 100 Index":  {"kind": "deriv", "symbol": "CRASH100",  "force": "SELL"},
    "Crash 200 Index":  {"kind": "deriv", "symbol": "CRASH200",  "force": "SELL"},
    "Crash 500 Index":  {"kind": "deriv", "symbol": "CRASH500",  "force": "SELL"},
    "Crash 900 Index":  {"kind": "deriv", "symbol": "CRASH900",  "force": "SELL"},
    "Crash 1000 Index": {"kind": "deriv", "symbol": "CRASH1000", "force": "SELL"},
}

DERIV_WS = "wss://ws.derivws.com/websockets/v3?app_id=1089"


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    return out.dropna()


def _make_buffers(h1: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build the M15/H1/H4/D1 buffer dict the tools expect from an H1 frame."""
    return {
        "M15": h1.tail(500).copy(),   # H1 used as M15 proxy when no lower TF
        "H1":  h1.tail(500).copy(),
        "H4":  _resample(h1, "4h").tail(500),
        "D1":  _resample(h1, "1D").tail(500),
    }


# --------------------------------------------------------------------------- #
#  yfinance source
# --------------------------------------------------------------------------- #
def _fetch_yf(ticker: str) -> Tuple[Dict[str, pd.DataFrame], float]:
    import yfinance as yf
    raw = yf.download(ticker, period="60d", interval="1h",
                      progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                   for c in raw.columns]
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    return _make_buffers(raw), float(raw["close"].iloc[-1])


# --------------------------------------------------------------------------- #
#  Deriv source
# --------------------------------------------------------------------------- #
def fetch_deriv_candles(symbol: str, granularity: int = 3600,
                        count: int = 500, retries: int = 3) -> pd.DataFrame:
    """Return an OHLCV DataFrame (UTC index) for a Deriv synthetic symbol."""
    import websocket  # websocket-client
    last_err = None
    for _ in range(retries):
        try:
            ws = websocket.create_connection(DERIV_WS, timeout=25)
            ws.send(json.dumps({
                "ticks_history": symbol, "end": "latest", "count": count,
                "style": "candles", "granularity": granularity,
            }))
            resp = json.loads(ws.recv())
            ws.close()
            if "candles" not in resp:
                raise RuntimeError(resp.get("error", {}).get("message", str(resp)))
            candles = resp["candles"]
            df = pd.DataFrame(candles)
            df["ts"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
            df = df.set_index("ts")[["open", "high", "low", "close"]].astype(float)
            df["volume"] = 0.0
            return df
        except Exception as e:      # noqa: BLE001
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"Deriv fetch failed for {symbol}: {last_err}")


def _fetch_deriv(symbol: str) -> Tuple[Dict[str, pd.DataFrame], float]:
    h1 = fetch_deriv_candles(symbol, granularity=3600, count=500)
    return _make_buffers(h1), float(h1["close"].iloc[-1])


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def load_instrument(name: str) -> Tuple[Dict[str, pd.DataFrame], float]:
    """Return (buffers, latest_price) for a registered instrument."""
    meta = INSTRUMENTS[name]
    if meta["kind"] == "yf":
        return _fetch_yf(meta["ticker"])
    return _fetch_deriv(meta["symbol"])


def latest_price(name: str) -> float:
    """Lightweight current-price fetch used by the performance tracker."""
    meta = INSTRUMENTS[name]
    if meta["kind"] == "yf":
        import yfinance as yf
        raw = yf.download(meta["ticker"], period="5d", interval="1h",
                          progress=False, auto_adjust=True)
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]
        return float(raw["close"].dropna().iloc[-1])
    return float(fetch_deriv_candles(meta["symbol"], 3600, 5)["close"].iloc[-1])


def price_path_since(name: str, since_epoch: int) -> pd.DataFrame:
    """
    OHLC candles from ~since_epoch to now, used to check whether TP or SL was
    touched first after a signal was issued.
    """
    meta = INSTRUMENTS[name]
    if meta["kind"] == "yf":
        import yfinance as yf
        raw = yf.download(meta["ticker"], period="30d", interval="1h",
                          progress=False, auto_adjust=True)
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]
        raw = raw[["open", "high", "low", "close"]].dropna()
        raw.index = pd.to_datetime(raw.index, utc=True)
        return raw[raw.index >= pd.to_datetime(since_epoch, unit="s", utc=True)]
    df = fetch_deriv_candles(meta["symbol"], 3600, 500)
    return df[df.index >= pd.to_datetime(since_epoch, unit="s", utc=True)]
