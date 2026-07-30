"""Utility functions shared by executable trader skills."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

TIMEFRAME_PRIORITY = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")


def clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(np.nan_to_num(value, nan=0.0, posinf=hi, neginf=lo), lo, hi))


def get_buffers(context: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    buffers = context.get("buffers") or context.get("market_buffers") or {}
    return buffers if isinstance(buffers, dict) else {}


def get_df(context_or_buffers: Dict[str, Any], preferred: Sequence[str] = ("M15", "H1", "M5", "M30", "D1")) -> Optional[pd.DataFrame]:
    buffers = get_buffers(context_or_buffers)
    if not buffers and all(col in context_or_buffers for col in ("open", "high", "low", "close")):
        return context_or_buffers  # type: ignore[return-value]
    for tf in preferred:
        df = buffers.get(tf)
        if is_ohlcv(df):
            return df.copy()
    for tf in TIMEFRAME_PRIORITY:
        df = buffers.get(tf)
        if is_ohlcv(df):
            return df.copy()
    for df in buffers.values():
        if is_ohlcv(df):
            return df.copy()
    return None


def is_ohlcv(df: Any) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and {"open", "high", "low", "close"}.issubset(df.columns)


def close(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["close"], errors="coerce").astype(float)


def high(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["high"], errors="coerce").astype(float)


def low(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["low"], errors="coerce").astype(float)


def open_(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["open"], errors="coerce").astype(float)


def volume(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(np.ones(len(df)), index=df.index, dtype=float)
    return pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float)


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=max(2, period // 3)).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=max(2, period // 3)).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    h = high(df)
    l = low(df)
    c = close(df)
    prev = c.shift(1)
    return pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period, min_periods=max(2, period // 3)).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=max(2, period // 3)).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=max(2, period // 3)).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    c = close(df)
    hh = high(df).rolling(period, min_periods=max(2, period // 3)).max()
    ll = low(df).rolling(period, min_periods=max(2, period // 3)).min()
    return 100.0 * (c - ll) / (hh - ll + 1e-12)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = typical_price(df)
    ma = tp.rolling(period, min_periods=max(2, period // 3)).mean()
    mad = (tp - ma).abs().rolling(period, min_periods=max(2, period // 3)).mean()
    return (tp - ma) / (0.015 * mad + 1e-12)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def obv(df: pd.DataFrame) -> pd.Series:
    c = close(df)
    v = volume(df)
    direction = np.sign(c.diff().fillna(0.0))
    return (direction * v).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = typical_price(df)
    v = volume(df).replace(0.0, np.nan)
    return (tp * v).cumsum() / (v.cumsum() + 1e-12)


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (high(df) + low(df) + close(df)) / 3.0


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    h, l, c, v = high(df), low(df), close(df), volume(df)
    mfm = ((c - l) - (h - c)) / (h - l + 1e-12)
    mfv = mfm * v
    return mfv.rolling(period, min_periods=max(2, period // 3)).sum() / (
        v.rolling(period, min_periods=max(2, period // 3)).sum() + 1e-12
    )


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = typical_price(df)
    raw = tp * volume(df)
    pos = raw.where(tp.diff() > 0, 0.0)
    neg = raw.where(tp.diff() < 0, 0.0).abs()
    ratio = pos.rolling(period, min_periods=max(2, period // 3)).sum() / (
        neg.rolling(period, min_periods=max(2, period // 3)).sum() + 1e-12
    )
    return 100.0 - (100.0 / (1.0 + ratio))


def linear_slope_score(series: pd.Series, lookback: int = 20) -> float:
    y = series.dropna().tail(lookback).to_numpy(dtype=float)
    if y.size < 3:
        return 0.0
    x = np.arange(y.size, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    denom = np.std(y) + 1e-12
    return clamp(slope * y.size / denom)


def pct_change_score(series: pd.Series, lookback: int = 20) -> float:
    y = series.dropna().tail(lookback)
    if len(y) < 2 or abs(float(y.iloc[0])) < 1e-12:
        return 0.0
    return clamp((float(y.iloc[-1]) - float(y.iloc[0])) / abs(float(y.iloc[0])) * 50.0)


def recent_atr_value(df: pd.DataFrame, period: int = 14) -> float:
    val = atr(df, period).dropna()
    if val.empty:
        tr = true_range(df).dropna()
        return float(tr.tail(20).mean()) if not tr.empty else 0.0
    return float(val.iloc[-1])


def rolling_range(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return high(df).rolling(period, min_periods=max(2, period // 3)).max() - low(df).rolling(
        period, min_periods=max(2, period // 3)
    ).min()


def swing_highs_lows(df: pd.DataFrame, window: int = 3) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    h, l = high(df).reset_index(drop=True), low(df).reset_index(drop=True)
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    for i in range(window, len(df) - window):
        hi = h.iloc[i]
        lo = l.iloc[i]
        if hi >= h.iloc[i - window : i + window + 1].max():
            highs.append((i, float(hi)))
        if lo <= l.iloc[i - window : i + window + 1].min():
            lows.append((i, float(lo)))
    return highs, lows


def direction_from_score(score: float, deadzone: float = 0.15) -> str:
    if score > deadzone:
        return "BULLISH"
    if score < -deadzone:
        return "BEARISH"
    return "NEUTRAL"


def latest_price(context: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> float:
    price = context.get("price")
    if isinstance(price, (int, float)) and price > 0:
        return float(price)
    snapshot = context.get("snapshot")
    tick = getattr(snapshot, "tick", None)
    if tick and getattr(tick, "bid", 0) and getattr(tick, "ask", 0):
        return float((tick.bid + tick.ask) / 2.0)
    if df is not None and is_ohlcv(df):
        return float(close(df).iloc[-1])
    return 0.0
