"""
data_feeds.py
=============
Fetches REAL market data for every instrument the Veilcrean signal engine
covers and returns multi-timeframe OHLCV buffers ready for the analysis tools.

Supports two modes:
  * "swing"  -> base H1 candles (slower, wider targets)
  * "scalp"  -> base M5 candles (fast, tight targets for quick trades)

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

import pandas as pd

# --------------------------------------------------------------------------- #
#  Instrument registry
# --------------------------------------------------------------------------- #
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

# base candle granularity per mode
DERIV_GRAN = {"swing": 3600, "scalp": 300}      # seconds
YF_INTERVAL = {"swing": ("60d", "1h"), "scalp": ("5d", "5m")}


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


def _make_buffers(base: pd.DataFrame, mode: str) -> Dict[str, pd.DataFrame]:
    """Build the M15/H1/H4/D1 buffer dict from a base frame.

    The fast "M15" slot always holds the base timeframe, so ATR-based targets
    scale naturally: hourly for swing, 5-minute for scalp.
    """
    if mode == "scalp":
        return {
            "M15": base.tail(500).copy(),                    # 5m
            "H1":  _resample(base, "15min").tail(500),       # 15m
            "H4":  _resample(base, "1h").tail(500),          # 1h
            "D1":  _resample(base, "4h").tail(500),          # 4h
        }
    return {
        "M15": base.tail(500).copy(),                        # 1h
        "H1":  base.tail(500).copy(),
        "H4":  _resample(base, "4h").tail(500),
        "D1":  _resample(base, "1D").tail(500),
    }


# --------------------------------------------------------------------------- #
#  yfinance source
# --------------------------------------------------------------------------- #
def _yf_download(ticker: str, mode: str) -> pd.DataFrame:
    import yfinance as yf
    period, interval = YF_INTERVAL[mode]
    raw = yf.download(ticker, period=period, interval=interval,
                      progress=False, auto_adjust=True)
    if (raw is None or raw.empty) and mode == "scalp":
        raw = yf.download(ticker, period="1mo", interval="15m",
                          progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                   for c in raw.columns]
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()
    raw.index = pd.to_datetime(raw.index, utc=True)
    return raw


def _fetch_yf(ticker: str, mode: str) -> Tuple[Dict[str, pd.DataFrame], float]:
    raw = _yf_download(ticker, mode)
    return _make_buffers(raw, mode), float(raw["close"].iloc[-1])


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
            df = pd.DataFrame(resp["candles"])
            df["ts"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
            df = df.set_index("ts")[["open", "high", "low", "close"]].astype(float)
            df["volume"] = 0.0
            return df
        except Exception as e:      # noqa: BLE001
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"Deriv fetch failed for {symbol}: {last_err}")


def _fetch_deriv(symbol: str, mode: str) -> Tuple[Dict[str, pd.DataFrame], float]:
    base = fetch_deriv_candles(symbol, granularity=DERIV_GRAN[mode], count=500)
    return _make_buffers(base, mode), float(base["close"].iloc[-1])


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def load_instrument(name: str, mode: str = "scalp"
                    ) -> Tuple[Dict[str, pd.DataFrame], float]:
    """Return (buffers, latest_price) for a registered instrument."""
    meta = INSTRUMENTS[name]
    if meta["kind"] == "yf":
        return _fetch_yf(meta["ticker"], mode)
    return _fetch_deriv(meta["symbol"], mode)


def latest_price(name: str) -> float:
    meta = INSTRUMENTS[name]
    if meta["kind"] == "yf":
        return float(_yf_download(meta["ticker"], "scalp")["close"].iloc[-1])
    return float(fetch_deriv_candles(meta["symbol"], 300, 5)["close"].iloc[-1])


def price_path_since(name: str, since_epoch: int,
                     mode: str = "scalp") -> pd.DataFrame:
    """OHLC candles from ~since_epoch to now, to check whether TP or SL was
    touched first after a signal was issued."""
    meta = INSTRUMENTS[name]
    if meta["kind"] == "yf":
        raw = _yf_download(meta["ticker"], mode)[["open", "high", "low", "close"]]
        return raw[raw.index >= pd.to_datetime(since_epoch, unit="s", utc=True)]
    df = fetch_deriv_candles(meta["symbol"], DERIV_GRAN[mode], 500)
    return df[df.index >= pd.to_datetime(since_epoch, unit="s", utc=True)]
