"""
Signal generator — runs lightweight technical analysis on candle data
and produces BUY/SELL/HOLD signals with confidence scores.

Implements 20+ indicators from scratch (numpy only) covering the same
categories as Veilcrean's 153 tools: market structure, momentum, volatility,
volume, trend, mean reversion, and pattern recognition. Each indicator
returns a score in [-1, 1] and the combined score determines the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Signal:
    """A trading signal with all indicator scores."""
    epoch: int
    direction: str           # BUY, SELL, HOLD
    confidence: float        # 0..1
    regime: str              # TRENDING, RANGING, VOLATILE, CHOPPY, BREAKOUT
    tool_scores: Dict[str, float] = field(default_factory=dict)
    combined_score: float = 0.0
    price: float = 0.0
    recommended_tp: float = 0.0  # pips
    recommended_sl: float = 0.0  # pips
    signal_rank: int = 0


# ============================= INDICATORS ============================= #

def _sma(series: np.ndarray, period: int) -> np.ndarray:
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    kernel = np.ones(period) / period
    return np.convolve(series, kernel, mode="same")


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    result = np.empty_like(series, dtype=float)
    result[:period - 1] = np.nan
    result[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(-period, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return float(np.mean(trs))


def _bollinger_position(closes: np.ndarray, period: int = 20) -> float:
    """Position within Bollinger Bands: -1 (lower band) to +1 (upper band)."""
    if len(closes) < period:
        return 0.0
    window = closes[-period:]
    mean = window.mean()
    std = window.std()
    if std == 0:
        return 0.0
    return float(np.clip((closes[-1] - mean) / (2 * std), -1, 1))


def _macd(closes: np.ndarray) -> tuple[float, float]:
    """Returns (macd line, signal line)."""
    if len(closes) < 35:
        return 0.0, 0.0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line[~np.isnan(macd_line)], 9) if np.any(~np.isnan(macd_line)) else np.array([0])
    return float(macd_line[-1]), float(signal_line[-1]) if len(signal_line) else 0.0


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> float:
    if len(closes) < period * 2:
        return 0.0
    plus_dm = []
    minus_dm = []
    tr_list = []
    for i in range(-period, 0):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0)
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        tr_list.append(tr)
    atr_val = np.mean(tr_list) if tr_list else 1
    if atr_val == 0:
        return 0.0
    plus_di = 100 * np.mean(plus_dm) / atr_val
    minus_di = 100 * np.mean(minus_dm) / atr_val
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    return float(dx)


def _stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> float:
    if len(closes) < period:
        return 50.0
    window_high = highs[-period:].max()
    window_low = lows[-period:].min()
    if window_high == window_low:
        return 50.0
    return float(100 * (closes[-1] - window_low) / (window_high - window_low))


def _williams_r(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 14) -> float:
    if len(closes) < period:
        return -50.0
    window_high = highs[-period:].max()
    window_low = lows[-period:].min()
    if window_high == window_low:
        return -50.0
    return float(-100 * (window_high - closes[-1]) / (window_high - window_low))


def _momentum(closes: np.ndarray, period: int = 10) -> float:
    if len(closes) < period + 1:
        return 0.0
    return float(closes[-1] - closes[-period - 1])


def _roc(closes: np.ndarray, period: int = 12) -> float:
    if len(closes) < period + 1 or closes[-period - 1] == 0:
        return 0.0
    return float(100 * (closes[-1] - closes[-period - 1]) / closes[-period - 1])


def _cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 20) -> float:
    if len(closes) < period:
        return 0.0
    tp = (highs[-period:] + lows[-period:] + closes[-period:]) / 3
    mean_tp = tp.mean()
    mean_dev = np.mean(np.abs(tp - mean_tp))
    if mean_dev == 0:
        return 0.0
    return float((tp[-1] - mean_tp) / (0.015 * mean_dev))


def _obv(closes: np.ndarray, volumes: np.ndarray) -> float:
    if len(closes) < 2:
        return 0.0
    obv = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i] if i < len(volumes) else 0
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i] if i < len(volumes) else 0
    return float(obv)


def _vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    if len(closes) == 0:
        return 0.0
    typical = (highs + lows + closes) / 3
    return float(typical.mean())


def _detect_breakout(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, lookback: int = 20) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    recent_high = highs[-(lookback + 1):-1].max()
    recent_low = lows[-(lookback + 1):-1].min()
    if closes[-1] > recent_high:
        return 1.0
    if closes[-1] < recent_low:
        return -1.0
    return 0.0


def _market_structure(closes: np.ndarray, lookback: int = 10) -> float:
    if len(closes) < lookback:
        return 0.0
    recent = closes[-lookback:]
    slope = np.polyfit(range(lookback), recent, 1)[0] if lookback > 1 else 0
    max_val = recent.max()
    min_val = recent.min()
    range_val = max_val - min_val
    if range_val == 0:
        return 0.0
    return float(np.clip(slope / (range_val / lookback), -1, 1))


def _volatility_regime(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, period: int = 20) -> str:
    atr_val = _atr(highs, lows, closes, period)
    if atr_val == 0 or len(closes) < period:
        return "UNKNOWN"
    avg_price = np.mean(closes[-period:])
    atr_pct = atr_val / avg_price if avg_price else 0

    # ADX for trend strength
    adx_val = _adx(highs, lows, closes, 14)

    if atr_pct > 0.01:
        if adx_val > 25:
            return "VOLATILE"
        return "CHOPPY"
    if adx_val > 25:
        return "TRENDING"
    if adx_val < 20 and atr_pct < 0.005:
        return "RANGING"
    if _detect_breakout(highs, lows, closes, 20) != 0:
        return "BREAKOUT"
    return "RANGING"


# ============================= SIGNAL GENERATION ============================= #

def generate_signal(candles: list[dict], pip_size: float = 0.0001,
                    min_history: int = 50) -> Optional[Signal]:
    """Generate a trading signal from candle data.

    `candles` should be sorted by epoch (oldest first), with the last
    candle being the most recent. Returns None if not enough data.
    """
    if len(candles) < min_history:
        return None

    # All implemented indicators use <= 50 bars of lookback, so keep a bounded
    # rolling analysis window while the outer trainer still walks every real
    # historical candle from oldest to newest. This makes full multi-run Deriv
    # training practical without changing indicator semantics.
    analysis_candles = candles[-80:]
    closes = np.array([c["close"] for c in analysis_candles], dtype=float)
    highs = np.array([c["high"] for c in analysis_candles], dtype=float)
    lows = np.array([c["low"] for c in analysis_candles], dtype=float)
    volumes = np.array([c.get("volume", 0) for c in analysis_candles], dtype=float)

    current_price = closes[-1]
    regime = _volatility_regime(highs, lows, closes)
    atr_val = _atr(highs, lows, closes, 14)

    # --- Compute all indicator scores ---
    scores: Dict[str, float] = {}

    # 1. RSI
    rsi_val = _rsi(closes, 14)
    scores["rsi"] = (50 - rsi_val) / 50  # >0 = oversold (buy), <0 = overbought (sell)

    # 2. MACD
    macd_line, signal_line = _macd(closes)
    scores["macd"] = float(np.clip(macd_line - signal_line, -1, 1))

    # 3. Bollinger position
    scores["bollinger"] = -_bollinger_position(closes, 20)  # below band = buy

    # 4. Stochastic
    stoch = _stochastic(highs, lows, closes, 14)
    scores["stochastic"] = (50 - stoch) / 50

    # 5. Williams %R
    wr = _williams_r(highs, lows, closes, 14)
    scores["williams_r"] = (wr + 50) / 50  # -100..0 -> -1..1 inverted

    # 6. Momentum
    scores["momentum"] = float(np.clip(_momentum(closes, 10) / (atr_val + 1e-10), -1, 1))

    # 7. Rate of Change
    scores["roc"] = float(np.clip(_roc(closes, 12) / 10, -1, 1))

    # 8. CCI
    cci_val = _cci(highs, lows, closes, 20)
    scores["cci"] = float(np.clip(-cci_val / 200, -1, 1))

    # 9. Market structure (trend slope)
    scores["market_structure"] = _market_structure(closes, 10)

    # 10. ADX strength
    adx_val = _adx(highs, lows, closes, 14)
    scores["adx"] = float(np.clip((adx_val - 20) / 30, 0, 1)) * np.sign(scores["market_structure"])

    # 11. Breakout detection
    scores["breakout"] = _detect_breakout(highs, lows, closes, 20)

    # 12. SMA crossover (fast vs slow)
    if len(closes) >= 50:
        sma_fast = _sma(closes, 20)[-1]
        sma_slow = _sma(closes, 50)[-1]
        if sma_slow > 0:
            scores["sma_cross"] = float(np.clip((sma_fast - sma_slow) / sma_slow * 100, -1, 1))
        else:
            scores["sma_cross"] = 0.0
    else:
        scores["sma_cross"] = 0.0

    # 13. EMA crossover
    if len(closes) >= 26:
        ema_fast = _ema(closes, 12)[-1]
        ema_slow = _ema(closes, 26)[-1]
        if not np.isnan(ema_slow) and ema_slow > 0:
            scores["ema_cross"] = float(np.clip((ema_fast - ema_slow) / ema_slow * 100, -1, 1))
        else:
            scores["ema_cross"] = 0.0
    else:
        scores["ema_cross"] = 0.0

    # 14. VWAP deviation
    vwap_val = _vwap(highs[-20:], lows[-20:], closes[-20:])
    if vwap_val > 0:
        scores["vwap_dev"] = float(np.clip((closes[-1] - vwap_val) / vwap_val * 100, -1, 1))
    else:
        scores["vwap_dev"] = 0.0

    # 15. ATR-based volatility (neutral, but informs TP/SL)
    scores["volatility"] = float(atr_val / current_price) if current_price > 0 else 0.0

    # 16. OBV trend
    obv_val = _obv(closes[-20:], volumes[-20:])
    scores["obv"] = float(np.clip(obv_val / (abs(obv_val) + 1), -1, 1))

    # 17. Higher high / lower low pattern
    if len(closes) >= 5:
        hh = closes[-1] > closes[-3] > closes[-5]
        ll = closes[-1] < closes[-3] < closes[-5]
        if hh:
            scores["pattern_hh"] = 0.5
        elif ll:
            scores["pattern_hh"] = -0.5
        else:
            scores["pattern_hh"] = 0.0
    else:
        scores["pattern_hh"] = 0.0

    # 18. Candle body
    if len(closes) >= 2:
        body = closes[-1] - analysis_candles[-1]["open"]
        scores["candle_body"] = float(np.clip(body / (atr_val + 1e-10), -1, 1))
    else:
        scores["candle_body"] = 0.0

    # 19. Wick rejection
    if len(highs) >= 1:
        upper_wick = highs[-1] - max(closes[-1], analysis_candles[-1]["open"])
        lower_wick = min(closes[-1], analysis_candles[-1]["open"]) - lows[-1]
        scores["wick_rejection"] = float(np.clip(
            (lower_wick - upper_wick) / (atr_val + 1e-10), -1, 1))
    else:
        scores["wick_rejection"] = 0.0

    # 20. Support/Resistance proximity
    if len(closes) >= 50:
        recent_high = highs[-50:].max()
        recent_low = lows[-50:].min()
        price_range = recent_high - recent_low
        if price_range > 0:
            pos = (closes[-1] - recent_low) / price_range
            # Near support = buy signal, near resistance = sell signal
            scores["sr_proximity"] = float(np.clip(0.5 - pos, -1, 1))
        else:
            scores["sr_proximity"] = 0.0
    else:
        scores["sr_proximity"] = 0.0

    # --- Combine scores ---
    # Weight directional indicators (those that suggest buy/sell)
    directional_weights = {
        "rsi": 1.0, "macd": 1.2, "bollinger": 0.8, "stochastic": 0.8,
        "williams_r": 0.6, "momentum": 1.0, "roc": 0.8, "cci": 0.8,
        "market_structure": 1.5, "adx": 0.8, "breakout": 1.5,
        "sma_cross": 1.0, "ema_cross": 1.0, "vwap_dev": 0.8,
        "obv": 0.6, "pattern_hh": 0.5, "candle_body": 0.7,
        "wick_rejection": 0.5, "sr_proximity": 0.8,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for key, weight in directional_weights.items():
        val = scores.get(key, 0.0)
        weighted_sum += val * weight
        total_weight += weight

    combined = weighted_sum / total_weight if total_weight > 0 else 0.0
    combined = float(np.clip(combined, -1, 1))

    # Direction
    if combined > 0.15:
        direction = "BUY"
    elif combined < -0.15:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Confidence: how strong is the signal?
    confidence = float(min(1.0, abs(combined) * 2))

    # Recommended TP/SL based on ATR
    atr_pips = atr_val / pip_size if pip_size > 0 else atr_val
    if direction == "BUY":
        recommended_tp = max(10, atr_pips * 1.5)
        recommended_sl = max(5, atr_pips * 0.8)
    elif direction == "SELL":
        recommended_tp = max(10, atr_pips * 1.5)
        recommended_sl = max(5, atr_pips * 0.8)
    else:
        recommended_tp = 0
        recommended_sl = 0

    return Signal(
        epoch=candles[-1]["epoch"],
        direction=direction,
        confidence=confidence,
        regime=regime,
        tool_scores=scores,
        combined_score=combined,
        price=current_price,
        recommended_tp=recommended_tp,
        recommended_sl=recommended_sl,
    )
