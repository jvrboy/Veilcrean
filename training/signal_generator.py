"""
Signal generator v2 — 35 indicators with regime-aware confidence.

Improvements over v1:
1. Added 15 new indicators (Ichimoku, ATR channels, multi-TF trend,
   Keltner, Hull MA, DMI, TTM Squeeze, volume profile, supply/demand zones)
2. Regime-aware indicator weighting
3. Dynamic TP/SL from learning engine
4. Confidence calibration per regime
5. Multi-timeframe alignment score
6. Supply/demand zone detection
7. Volume-confirmation filter
"""
from __future__ import annotations

import numpy as np
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
    if len(closes) < period:
        return 0.0
    window = closes[-period:]
    mean = window.mean()
    std = window.std()
    if std == 0:
        return 0.0
    return float(np.clip((closes[-1] - mean) / (2 * std), -1, 1))


def _bollinger_width(closes: np.ndarray, period: int = 20) -> float:
    """Bollinger Band width normalized — high = volatile, low = squeeze."""
    if len(closes) < period:
        return 0.0
    std = closes[-period:].std()
    mean = closes[-period:].mean()
    return float(std / mean) if mean > 0 else 0.0


def _macd(closes: np.ndarray) -> tuple[float, float]:
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


def _dmi_direction(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    period: int = 14) -> float:
    """DMI directional bias: +1 = bullish, -1 = bearish."""
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
    if plus_di + minus_di == 0:
        return 0.0
    return float((plus_di - minus_di) / (plus_di + minus_di))


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


# ---- NEW INDICATORS (v2) ----

def _hull_ma_trend(closes: np.ndarray) -> float:
    """Hull MA trend direction: +1 bullish, -1 bearish."""
    if len(closes) < 40:
        return 0.0
    half_period = 10
    wma_half = _ema(closes, half_period)
    full_period = 20
    wma_full = _ema(closes, full_period)
    valid = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    if not np.any(valid):
        return 0.0
    hull = 2 * wma_half[valid] - wma_full[valid]
    if len(hull) < 2:
        return 0.0
    return float(np.clip((hull[-1] - hull[-5]) / (abs(hull[-1]) + 1e-10), -1, 1))


def _keltner_position(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """Position within Keltner Channels."""
    if len(closes) < 20:
        return 0.0
    ema_val = _ema(closes, 20)
    atr_val = _atr(highs, lows, closes, 20)
    if atr_val == 0:
        return 0.0
    upper = ema_val[-1] + 2 * atr_val
    lower = ema_val[-1] - 2 * atr_val
    if upper == lower:
        return 0.0
    return float(np.clip(2 * (closes[-1] - ema_val[-1]) / (upper - lower), -1, 1))


def _ttm_squeeze(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """TTM Squeeze: +1 = squeeze on (expansion coming), -1 = no squeeze."""
    if len(closes) < 30:
        return 0.0
    bb_width = _bollinger_width(closes, 20)
    keltner_width = _atr(highs, lows, closes, 20) / closes[-1] if closes[-1] > 0 else 0
    # Squeeze is on when BB is inside Keltner
    if bb_width < keltner_width * 1.5:
        return 0.8  # Squeeze on — expect expansion
    return -0.3  # No squeeze


def _multi_tf_trend(closes: np.ndarray) -> float:
    """Multi-timeframe trend alignment from 3 EMAs."""
    if len(closes) < 50:
        return 0.0
    e8 = _ema(closes, 8)[-1] if not np.isnan(_ema(closes, 8)[-1]) else closes[-1]
    e21 = _ema(closes, 21)[-1] if not np.isnan(_ema(closes, 21)[-1]) else closes[-1]
    e50 = _ema(closes, 50)[-1] if not np.isnan(_ema(closes, 50)[-1]) else closes[-1]
    if e8 > e21 > e50:
        return 1.0
    if e8 < e21 < e50:
        return -1.0
    if e8 > e21 or e21 > e50:
        return 0.3
    if e8 < e21 or e21 < e50:
        return -0.3
    return 0.0


def _supply_demand_zone(closes: np.ndarray, highs: np.ndarray,
                         lows: np.ndarray) -> float:
    """Supply/demand zone proximity: +1 near demand, -1 near supply."""
    if len(closes) < 40:
        return 0.0
    # Simple S/D: recent swing lows as demand, swing highs as supply
    recent_lows = lows[-40:]
    recent_highs = highs[-40:]
    # Find demand zone (cluster of lows)
    low_zones = sorted(recent_lows)[:5]
    demand_level = np.mean(low_zones)
    # Find supply zone (cluster of highs)
    high_zones = sorted(recent_highs)[-5:]
    supply_level = np.mean(high_zones)
    price = closes[-1]
    range_val = supply_level - demand_level
    if range_val == 0:
        return 0.0
    # Position: 0 at supply, 1 at demand
    pos = (supply_level - price) / range_val
    return float(np.clip(2 * (pos - 0.5), -1, 1))


def _chop_index(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    """Choppiness index: <38.2 = trending, >61.8 = choppy."""
    if len(closes) < 14:
        return 0.5
    trs = []
    for i in range(-14, 0):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        trs.append(tr)
    sum_tr = sum(trs)
    high_max = highs[-14:].max()
    low_min = lows[-14:].min()
    price_range = high_max - low_min
    if sum_tr == 0 or price_range == 0:
        return 0.5
    chop = 100 * np.log10(sum_tr / price_range) / np.log10(14)
    # Convert to -1 (choppy) to +1 (trending)
    return float(np.clip((61.8 - chop) / 23.6, -1, 1))


def _awesome_oscillator(highs: np.ndarray, lows: np.ndarray) -> float:
    """Awesome Oscillator: midpoint of 5-period vs 34-period."""
    if len(highs) < 34:
        return 0.0
    mid_5 = ((highs[-5:] + lows[-5:]) / 2).mean()
    mid_34 = ((highs[-34:] + lows[-34:]) / 2).mean()
    if mid_34 == 0:
        return 0.0
    return float(np.clip((mid_5 - mid_34) / mid_34 * 100, -1, 1))


def _ichimoku_cloud(closes: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray) -> float:
    """Ichimoku cloud position: +1 above cloud, -1 below cloud."""
    if len(closes) < 52:
        return 0.0
    # Tenkan-sen (9-period)
    tenkan = (highs[-9:].max() + lows[-9:].min()) / 2
    # Kijun-sen (26-period)
    kijun = (highs[-26:].max() + lows[-26:].min()) / 2
    # Senkou Span A (average of tenkan/kijun, shifted 26 ahead)
    span_a = (tenkan + kijun) / 2
    # Senkou Span B (52-period, shifted 26 ahead)
    span_b = (highs[-52:].max() + lows[-52:].min()) / 2
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)
    price = closes[-1]
    if cloud_top == cloud_bottom:
        return 0.0
    # Position within cloud: -1 below, 0 in cloud, +1 above
    if price > cloud_top:
        return 1.0
    elif price < cloud_bottom:
        return -1.0
    else:
        # Inside cloud — bias toward closer edge
        pos = (price - cloud_bottom) / (cloud_top - cloud_bottom)
        return float(np.clip(2 * (pos - 0.5), -0.5, 0.5))


def _mfi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         volumes: np.ndarray, period: int = 14) -> float:
    """Money Flow Index — volume-weighted RSI."""
    if len(closes) < period + 1 or len(volumes) < period + 1:
        return 50.0
    typical = (highs + lows + closes) / 3
    raw_mf = typical * volumes
    deltas = np.diff(typical[-(period + 1):])
    pos_mf = np.where(deltas > 0, raw_mf[-(period):], 0).sum()
    neg_mf = np.where(deltas < 0, raw_mf[-(period):], 0).sum()
    if neg_mf == 0:
        return 100.0
    mfi = 100 - 100 / (1 + pos_mf / neg_mf)
    return float(mfi)


def _stochastic_rsi(closes: np.ndarray, rsi_period: int = 14,
                     stoch_period: int = 14) -> float:
    """Stochastic RSI — more sensitive than RSI."""
    if len(closes) < rsi_period + stoch_period:
        return 50.0
    rsi_values = []
    for i in range(len(closes) - rsi_period, len(closes)):
        if i < rsi_period + 1:
            rsi_values.append(50.0)
        else:
            rsi_values.append(_rsi(closes[:i+1], rsi_period))
    rsi_arr = np.array(rsi_values[-stoch_period:])
    rsi_min = rsi_arr.min()
    rsi_max = rsi_arr.max()
    if rsi_max == rsi_min:
        return 50.0
    return float(100 * (rsi_arr[-1] - rsi_min) / (rsi_max - rsi_min))


def _rsi_divergence(closes: np.ndarray, period: int = 14) -> float:
    """Detect RSI divergence: +1 bullish div, -1 bearish div."""
    if len(closes) < period * 3:
        return 0.0
    # Recent price action
    recent_price_high = closes[-period:].max()
    recent_price_low = closes[-period:].min()
    prior_price_high = closes[-period*2:-period].max()
    prior_price_low = closes[-period*2:-period].min()
    # Recent RSI
    rsi_now = _rsi(closes, period)
    rsi_prior = _rsi(closes[:-period] if len(closes) > period else closes, period)
    # Bullish divergence: price lower low, RSI higher low
    if recent_price_low < prior_price_low and rsi_now > rsi_prior:
        return 0.8
    # Bearish divergence: price higher high, RSI lower high
    if recent_price_high > prior_price_high and rsi_now < rsi_prior:
        return -0.8
    return 0.0


def _atr_channel_position(closes: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, period: int = 20) -> float:
    """Position within ATR channel."""
    if len(closes) < period:
        return 0.0
    atr_val = _atr(highs, lows, closes, period)
    mid = _ema(closes, period)[-1]
    if not np.isnan(mid) and atr_val > 0:
        upper = mid + 3 * atr_val
        lower = mid - 3 * atr_val
        if upper == lower:
            return 0.0
        return float(np.clip((closes[-1] - mid) / (upper - lower) * 2, -1, 1))
    return 0.0


def _volume_confirmation(closes: np.ndarray, volumes: np.ndarray) -> float:
    """Volume confirms price direction."""
    if len(closes) < 20 or len(volumes) < 20:
        return 0.0
    price_change = closes[-1] - closes[-20]
    avg_volume = volumes[-20:].mean()
    current_volume = volumes[-1]
    if avg_volume == 0:
        return 0.0
    vol_ratio = current_volume / avg_volume
    # High volume + price up = bullish, high volume + price down = bearish
    direction = 1 if price_change > 0 else -1
    return float(np.clip(direction * (vol_ratio - 1), -1, 1))


def _volatility_regime(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, period: int = 20) -> str:
    atr_val = _atr(highs, lows, closes, period)
    if atr_val == 0 or len(closes) < period:
        return "UNKNOWN"
    avg_price = np.mean(closes[-period:])
    atr_pct = atr_val / avg_price if avg_price else 0
    adx_val = _adx(highs, lows, closes, 14)
    chop = _chop_index(highs, lows, closes)

    if atr_pct > 0.01:
        if adx_val > 25:
            return "VOLATILE"
        return "CHOPPY"
    if adx_val > 25 or chop > 0.3:
        return "TRENDING"
    if chop < -0.3:
        return "CHOPPY"
    if _detect_breakout(highs, lows, closes, 20) != 0:
        return "BREAKOUT"
    return "RANGING"


# ============================= REGIME-AWARE WEIGHTS ============================= #

# Weights optimized per regime
REGIME_WEIGHTS = {
    "TRENDING": {
        "rsi": 0.8, "macd": 1.3, "bollinger": 0.6, "stochastic": 0.6,
        "williams_r": 0.5, "momentum": 1.2, "roc": 0.8, "cci": 0.7,
        "market_structure": 1.8, "adx": 1.2, "breakout": 1.5,
        "sma_cross": 1.2, "ema_cross": 1.2, "vwap_dev": 0.6,
        "obv": 0.5, "pattern_hh": 0.6, "candle_body": 0.8,
        "wick_rejection": 0.4, "sr_proximity": 0.7,
        # v2 indicators
        "hull_ma": 1.5, "keltner": 0.7, "dmi_direction": 1.3,
        "multi_tf_trend": 2.0, "ichimoku": 1.8, "mfi": 0.6,
        "stochastic_rsi": 0.5, "rsi_divergence": 1.0,
        "atr_channel": 0.6, "chop_index": 1.0, "awesome_osc": 0.7,
        "volume_conf": 0.7, "ttm_squeeze": 0.8, "supply_demand": 1.0,
    },
    "RANGING": {
        "rsi": 1.3, "macd": 0.6, "bollinger": 1.5, "stochastic": 1.5,
        "williams_r": 1.2, "momentum": 0.5, "roc": 0.5, "cci": 1.2,
        "market_structure": 0.5, "adx": 0.3, "breakout": 0.3,
        "sma_cross": 0.5, "ema_cross": 0.5, "vwap_dev": 0.8,
        "obv": 0.7, "pattern_hh": 0.3, "candle_body": 0.5,
        "wick_rejection": 1.2, "sr_proximity": 1.8,
        "hull_ma": 0.5, "keltner": 1.5, "dmi_direction": 0.3,
        "multi_tf_trend": 0.3, "ichimoku": 0.5, "mfi": 1.3,
        "stochastic_rsi": 1.5, "rsi_divergence": 0.8,
        "atr_channel": 0.8, "chop_index": 1.2, "awesome_osc": 0.5,
        "volume_conf": 0.5, "ttm_squeeze": 1.2, "supply_demand": 1.8,
    },
    "VOLATILE": {
        "rsi": 0.7, "macd": 0.8, "bollinger": 1.0, "stochastic": 0.7,
        "williams_r": 0.6, "momentum": 0.8, "roc": 0.6, "cci": 1.0,
        "market_structure": 0.8, "adx": 1.5, "breakout": 1.8,
        "sma_cross": 0.6, "ema_cross": 0.6, "vwap_dev": 0.5,
        "obv": 0.5, "pattern_hh": 0.4, "candle_body": 1.0,
        "wick_rejection": 1.0, "sr_proximity": 0.5,
        "hull_ma": 0.8, "keltner": 1.2, "dmi_direction": 0.8,
        "multi_tf_trend": 0.8, "ichimoku": 0.8, "mfi": 0.8,
        "stochastic_rsi": 0.7, "rsi_divergence": 0.6,
        "atr_channel": 1.5, "chop_index": 0.8, "awesome_osc": 0.6,
        "volume_conf": 1.0, "ttm_squeeze": 1.5, "supply_demand": 0.6,
    },
    "CHOPPY": {
        "rsi": 0.5, "macd": 0.3, "bollinger": 0.8, "stochastic": 0.8,
        "williams_r": 0.6, "momentum": 0.3, "roc": 0.3, "cci": 0.6,
        "market_structure": 0.3, "adx": 0.2, "breakout": 0.5,
        "sma_cross": 0.3, "ema_cross": 0.3, "vwap_dev": 0.6,
        "obv": 0.5, "pattern_hh": 0.2, "candle_body": 0.4,
        "wick_rejection": 1.5, "sr_proximity": 1.5,
        "hull_ma": 0.3, "keltner": 1.0, "dmi_direction": 0.3,
        "multi_tf_trend": 0.3, "ichimoku": 0.3, "mfi": 0.6,
        "stochastic_rsi": 0.8, "rsi_divergence": 0.5,
        "atr_channel": 0.8, "chop_index": 2.0, "awesome_osc": 0.3,
        "volume_conf": 0.4, "ttm_squeeze": 1.0, "supply_demand": 1.2,
    },
    "BREAKOUT": {
        "rsi": 0.6, "macd": 1.0, "bollinger": 0.8, "stochastic": 0.5,
        "williams_r": 0.4, "momentum": 1.5, "roc": 1.2, "cci": 0.8,
        "market_structure": 1.5, "adx": 1.5, "breakout": 2.0,
        "sma_cross": 1.0, "ema_cross": 1.0, "vwap_dev": 0.6,
        "obv": 0.7, "pattern_hh": 0.8, "candle_body": 1.2,
        "wick_rejection": 0.5, "sr_proximity": 1.0,
        "hull_ma": 1.2, "keltner": 0.8, "dmi_direction": 1.0,
        "multi_tf_trend": 1.5, "ichimoku": 1.5, "mfi": 0.7,
        "stochastic_rsi": 0.5, "rsi_divergence": 0.7,
        "atr_channel": 1.0, "chop_index": 0.5, "awesome_osc": 1.0,
        "volume_conf": 1.5, "ttm_squeeze": 1.8, "supply_demand": 0.8,
    },
    "UNKNOWN": {  # fallback
        "rsi": 1.0, "macd": 1.0, "bollinger": 1.0, "stochastic": 1.0,
        "williams_r": 0.6, "momentum": 1.0, "roc": 0.8, "cci": 0.8,
        "market_structure": 1.0, "adx": 0.8, "breakout": 1.0,
        "sma_cross": 1.0, "ema_cross": 1.0, "vwap_dev": 0.8,
        "obv": 0.6, "pattern_hh": 0.5, "candle_body": 0.7,
        "wick_rejection": 0.5, "sr_proximity": 0.8,
        "hull_ma": 0.8, "keltner": 0.8, "dmi_direction": 0.8,
        "multi_tf_trend": 0.8, "ichimoku": 0.8, "mfi": 0.8,
        "stochastic_rsi": 0.8, "rsi_divergence": 0.8,
        "atr_channel": 0.8, "chop_index": 0.8, "awesome_osc": 0.7,
        "volume_conf": 0.7, "ttm_squeeze": 0.8, "supply_demand": 0.8,
    },
}


# ============================= SIGNAL GENERATION ============================= #

def generate_signal(candles: list[dict], pip_size: float = 0.0001,
                    min_history: int = 50,
                    tp_override: float = 0.0,
                    sl_override: float = 0.0,
                    trailing_enabled: bool = False,
                    breakeven_trigger: float = 0.7) -> Optional[Signal]:
    """Generate a trading signal from candle data (v2).

    Supports optional TP/SL overrides from the learning engine.
    """
    if len(candles) < min_history:
        return None

    analysis_candles = candles[-80:]
    closes = np.array([c["close"] for c in analysis_candles], dtype=float)
    highs = np.array([c["high"] for c in analysis_candles], dtype=float)
    lows = np.array([c["low"] for c in analysis_candles], dtype=float)
    volumes = np.array([c.get("volume", 0) for c in analysis_candles], dtype=float)

    current_price = closes[-1]
    regime = _volatility_regime(highs, lows, closes)
    atr_val = _atr(highs, lows, closes, 14)
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["UNKNOWN"])

    # --- Compute all indicator scores ---
    scores: Dict[str, float] = {}

    # 1. RSI
    rsi_val = _rsi(closes, 14)
    scores["rsi"] = (50 - rsi_val) / 50

    # 2. MACD
    macd_line, signal_line = _macd(closes)
    scores["macd"] = float(np.clip(macd_line - signal_line, -1, 1))

    # 3. Bollinger position
    scores["bollinger"] = -_bollinger_position(closes, 20)

    # 4. Stochastic
    stoch = _stochastic(highs, lows, closes, 14)
    scores["stochastic"] = (50 - stoch) / 50

    # 5. Williams %R
    wr = _williams_r(highs, lows, closes, 14)
    scores["williams_r"] = (wr + 50) / 50

    # 6. Momentum
    scores["momentum"] = float(np.clip(_momentum(closes, 10) / (atr_val + 1e-10), -1, 1))

    # 7. Rate of Change
    scores["roc"] = float(np.clip(_roc(closes, 12) / 10, -1, 1))

    # 8. CCI
    cci_val = _cci(highs, lows, closes, 20)
    scores["cci"] = float(np.clip(-cci_val / 200, -1, 1))

    # 9. Market structure
    scores["market_structure"] = _market_structure(closes, 10)

    # 10. ADX strength
    adx_val = _adx(highs, lows, closes, 14)
    scores["adx"] = float(np.clip((adx_val - 20) / 30, 0, 1)) * np.sign(scores["market_structure"])

    # 11. Breakout
    scores["breakout"] = _detect_breakout(highs, lows, closes, 20)

    # 12. SMA crossover
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

    # 15. Volatility (neutral, for TP/SL)
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
            scores["sr_proximity"] = float(np.clip(0.5 - pos, -1, 1))
        else:
            scores["sr_proximity"] = 0.0
    else:
        scores["sr_proximity"] = 0.0

    # ===== V2 INDICATORS =====

    # 21. Hull MA trend
    scores["hull_ma"] = _hull_ma_trend(closes)

    # 22. Keltner position
    scores["keltner"] = _keltner_position(highs, lows, closes)

    # 23. TTM Squeeze
    scores["ttm_squeeze"] = _ttm_squeeze(highs, lows, closes)

    # 24. Multi-TF trend
    scores["multi_tf_trend"] = _multi_tf_trend(closes)

    # 25. Supply/Demand zone
    scores["supply_demand"] = _supply_demand_zone(closes, highs, lows)

    # 26. Chop Index
    scores["chop_index"] = _chop_index(highs, lows, closes)

    # 27. Awesome Oscillator
    scores["awesome_osc"] = _awesome_oscillator(highs, lows)

    # 28. Ichimoku Cloud
    scores["ichimoku"] = _ichimoku_cloud(closes, highs, lows)

    # 29. MFI
    mfi_val = _mfi(highs, lows, closes, volumes)
    scores["mfi"] = (50 - mfi_val) / 50

    # 30. Stochastic RSI
    stoch_rsi = _stochastic_rsi(closes)
    scores["stochastic_rsi"] = (50 - stoch_rsi) / 50

    # 31. RSI Divergence
    scores["rsi_divergence"] = _rsi_divergence(closes)

    # 32. ATR Channel
    scores["atr_channel"] = _atr_channel_position(closes, highs, lows)

    # 33. Volume Confirmation
    scores["volume_conf"] = _volume_confirmation(closes, volumes)

    # 34. DMI Direction
    scores["dmi_direction"] = _dmi_direction(highs, lows, closes)

    # --- Combine scores with regime-aware weights ---
    weighted_sum = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        if key == "volatility":
            continue  # Neutral, skip from direction
        val = scores.get(key, 0.0)
        weighted_sum += val * weight
        total_weight += weight

    combined = weighted_sum / total_weight if total_weight > 0 else 0.0
    combined = float(np.clip(combined, -1, 1))

    # Direction — higher threshold in choppy markets
    dir_threshold = 0.12 if regime == "TRENDING" or regime == "BREAKOUT" else 0.18
    if combined > dir_threshold:
        direction = "BUY"
    elif combined < -dir_threshold:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Confidence: regime-aware calibration
    if direction == "HOLD":
        confidence = 0.0
    else:
        # Base confidence from combined score strength
        confidence = float(min(1.0, abs(combined) * 2.2))
        # Boost confidence in favorable regimes
        if regime in ("TRENDING", "BREAKOUT"):
            confidence = min(1.0, confidence * 1.15)
        # Reduce confidence in unfavorable regimes
        elif regime in ("CHOPPY",):
            confidence *= 0.8
        # Boost if multiple strong indicators agree
        strong_count = sum(1 for v in scores.values() if abs(v) > 0.4 and v * combined > 0)
        if strong_count >= 5:
            confidence = min(1.0, confidence * 1.1)

    # TP/SL
    atr_pips = atr_val / pip_size if pip_size > 0 else atr_val
    if tp_override > 0 and sl_override > 0:
        recommended_tp = tp_override
        recommended_sl = sl_override
    elif direction != "HOLD":
        # Use regime-based defaults (will be overridden by learning engine)
        defaults = {
            "TRENDING": (2.0, 1.0),
            "RANGING": (1.2, 0.8),
            "VOLATILE": (1.5, 1.2),
            "CHOPPY": (0.8, 0.5),
            "BREAKOUT": (2.5, 1.0),
            "UNKNOWN": (1.5, 1.0),
        }
        tp_m, sl_m = defaults.get(regime, (1.5, 1.0))
        recommended_tp = max(5, atr_pips * tp_m)
        recommended_sl = max(3, atr_pips * sl_m)
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
