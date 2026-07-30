"""Executable trading skills used by agents and analysis tools."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .catalog import ALL_SKILL_DEFINITIONS
from .core import BaseSkill, SkillDefinition
from .market_utils import (
    atr,
    cci,
    clamp,
    close,
    cmf,
    direction_from_score,
    ema,
    get_buffers,
    get_df,
    high,
    is_ohlcv,
    latest_price,
    linear_slope_score,
    low,
    macd,
    mfi,
    obv,
    open_,
    pct_change_score,
    recent_atr_value,
    rolling_range,
    rsi,
    sma,
    stochastic,
    swing_highs_lows,
    true_range,
    typical_price,
    volume,
    vwap,
)

_DEF_MAP: Dict[str, SkillDefinition] = {d.id: d for d in ALL_SKILL_DEFINITIONS}


def _definition(skill_id: str) -> SkillDefinition:
    return _DEF_MAP[skill_id]


def _clean_features(features: Dict[str, Any]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for key, value in features.items():
        try:
            clean[key] = float(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0))
        except Exception:
            clean[key] = 0.0
    return clean


class TimeframeAnalysisSkill(BaseSkill):
    definition = _definition("ta_timeframe_analysis")

    def run(self, context: Dict[str, Any]):
        buffers = get_buffers(context)
        scores: Dict[str, float] = {}
        for tf, df in buffers.items():
            if is_ohlcv(df) and len(df) >= 8:
                scores[str(tf)] = linear_slope_score(close(df), lookback=min(50, max(8, len(df) // 3)))
        if not scores:
            return self.unavailable("no OHLCV buffers available")
        htf_keys = [tf for tf in ("H1", "H4", "D1", "W1") if tf in scores] or list(scores)
        htf_score = float(np.mean([scores[k] for k in htf_keys]))
        signs = [np.sign(v) for v in scores.values() if abs(v) > 0.1]
        alignment = float(max(signs.count(1.0), signs.count(-1.0)) / max(len(signs), 1)) if signs else 0.0
        score = clamp(htf_score)
        return self.result(
            score=score,
            confidence=0.35 + 0.55 * alignment,
            direction=direction_from_score(score),
            features=_clean_features({"mta_score": score, "mta_alignment": alignment, **{f"tf_{k}_slope": v for k, v in scores.items()}}),
            metadata={"timeframe_scores": scores, "higher_timeframe_keys": htf_keys},
            signals=[f"{direction_from_score(score)} multi-timeframe bias", f"alignment={alignment:.2f}"],
        )


class CandlestickPatternSkill(BaseSkill):
    definition = _definition("ta_candlestick_patterns")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("M5", "M15", "M30", "H1"))
        if df is None or len(df) < 3:
            return self.unavailable("need at least 3 OHLC candles")
        o, h, l, c = open_(df), high(df), low(df), close(df)
        body = (c - o).abs()
        rng = (h - l).replace(0.0, np.nan).abs()
        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - l
        i = -1
        patterns: Dict[str, int] = {}
        body_ratio = float((body / (rng + 1e-12)).iloc[i])
        upper_ratio = float((upper / (rng + 1e-12)).iloc[i])
        lower_ratio = float((lower / (rng + 1e-12)).iloc[i])
        is_bull = c.iloc[i] > o.iloc[i]
        prev_bull = c.iloc[-2] > o.iloc[-2]

        patterns["doji"] = int(body_ratio < 0.12)
        patterns["hammer"] = int(lower_ratio > 0.55 and upper_ratio < 0.25 and body_ratio < 0.35)
        patterns["shooting_star"] = int(upper_ratio > 0.55 and lower_ratio < 0.25 and body_ratio < 0.35)
        patterns["marubozu_bull"] = int(is_bull and body_ratio > 0.78)
        patterns["marubozu_bear"] = int((not is_bull) and body_ratio > 0.78)
        patterns["spinning_top"] = int(0.12 <= body_ratio < 0.28 and upper_ratio > 0.25 and lower_ratio > 0.25)

        engulf_bull = (not prev_bull) and is_bull and o.iloc[-1] <= c.iloc[-2] and c.iloc[-1] >= o.iloc[-2]
        engulf_bear = prev_bull and (not is_bull) and o.iloc[-1] >= c.iloc[-2] and c.iloc[-1] <= o.iloc[-2]
        patterns["bullish_engulfing"] = int(engulf_bull)
        patterns["bearish_engulfing"] = int(engulf_bear)
        patterns["harami"] = int(max(o.iloc[-1], c.iloc[-1]) < max(o.iloc[-2], c.iloc[-2]) and min(o.iloc[-1], c.iloc[-1]) > min(o.iloc[-2], c.iloc[-2]))

        three = df.tail(3)
        t_o, t_c = open_(three), close(three)
        patterns["three_white_soldiers"] = int(all(t_c > t_o) and t_c.is_monotonic_increasing)
        patterns["three_black_crows"] = int(all(t_c < t_o) and t_c.is_monotonic_decreasing)
        patterns["morning_star"] = int(t_c.iloc[0] < t_o.iloc[0] and abs(t_c.iloc[1] - t_o.iloc[1]) < abs(t_c.iloc[0] - t_o.iloc[0]) * 0.5 and t_c.iloc[2] > t_o.iloc[2] and t_c.iloc[2] > (t_o.iloc[0] + t_c.iloc[0]) / 2)
        patterns["evening_star"] = int(t_c.iloc[0] > t_o.iloc[0] and abs(t_c.iloc[1] - t_o.iloc[1]) < abs(t_c.iloc[0] - t_o.iloc[0]) * 0.5 and t_c.iloc[2] < t_o.iloc[2] and t_c.iloc[2] < (t_o.iloc[0] + t_c.iloc[0]) / 2)

        bull_score = patterns["hammer"] + patterns["marubozu_bull"] + patterns["bullish_engulfing"] + patterns["three_white_soldiers"] + patterns["morning_star"]
        bear_score = patterns["shooting_star"] + patterns["marubozu_bear"] + patterns["bearish_engulfing"] + patterns["three_black_crows"] + patterns["evening_star"]
        score = clamp((bull_score - bear_score) / 3.0)
        found = [name for name, active in patterns.items() if active]
        confidence = min(1.0, 0.35 + 0.15 * len(found))
        return self.result(
            score=score,
            confidence=confidence,
            direction=direction_from_score(score),
            features=_clean_features(patterns | {"body_ratio": body_ratio, "upper_wick_ratio": upper_ratio, "lower_wick_ratio": lower_ratio}),
            metadata={"patterns": found},
            signals=found or ["no decisive candlestick pattern"],
        )


class ReversalPatternSkill(BaseSkill):
    definition = _definition("ta_reversal_patterns")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("H1", "H4", "M15", "D1"))
        if df is None or len(df) < 40:
            return self.unavailable("need at least 40 candles for reversal patterns")
        highs, lows = swing_highs_lows(df, window=3)
        price = latest_price(context, df)
        atr_val = max(recent_atr_value(df), price * 0.001, 1e-12)
        signals: list[str] = []
        score = 0.0
        features: Dict[str, float] = {}

        if len(highs) >= 2:
            h1, h2 = highs[-2], highs[-1]
            equal = abs(h1[1] - h2[1]) <= 0.75 * atr_val
            features["double_top_quality"] = float(equal)
            if equal and price < min(h1[1], h2[1]) - 0.25 * atr_val:
                score -= 0.45
                signals.append("double top pressure")
        if len(lows) >= 2:
            l1, l2 = lows[-2], lows[-1]
            equal = abs(l1[1] - l2[1]) <= 0.75 * atr_val
            features["double_bottom_quality"] = float(equal)
            if equal and price > max(l1[1], l2[1]) + 0.25 * atr_val:
                score += 0.45
                signals.append("double bottom pressure")
        if len(highs) >= 3:
            a, b, c_ = highs[-3], highs[-2], highs[-1]
            hs = b[1] > a[1] and b[1] > c_[1] and abs(a[1] - c_[1]) <= 1.25 * atr_val
            features["head_shoulders_quality"] = float(hs)
            if hs:
                score -= 0.35
                signals.append("head and shoulders structure")
        if len(lows) >= 3:
            a, b, c_ = lows[-3], lows[-2], lows[-1]
            inv = b[1] < a[1] and b[1] < c_[1] and abs(a[1] - c_[1]) <= 1.25 * atr_val
            features["inverse_head_shoulders_quality"] = float(inv)
            if inv:
                score += 0.35
                signals.append("inverse head and shoulders structure")
        features["swing_high_count"] = len(highs)
        features["swing_low_count"] = len(lows)
        return self.result(score=clamp(score), confidence=0.45 + min(0.4, 0.08 * len(signals)), direction=direction_from_score(score), features=_clean_features(features), metadata={"signals": signals}, signals=signals or ["no major reversal pattern"])


class ContinuationPatternSkill(BaseSkill):
    definition = _definition("ta_continuation_patterns")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("M15", "H1", "M5", "H4"))
        if df is None or len(df) < 45:
            return self.unavailable("need at least 45 candles for continuation patterns")
        c = close(df)
        trend = linear_slope_score(c.iloc[:-12], lookback=30)
        rr = rolling_range(df, 10).dropna()
        compression = float(rr.tail(8).mean() / (rr.tail(30).mean() + 1e-12)) if len(rr) >= 30 else 1.0
        recent_slope = linear_slope_score(c, lookback=12)
        flag_like = abs(trend) > 0.35 and compression < 0.75 and np.sign(recent_slope) != np.sign(trend)
        triangle_like = compression < 0.65
        breakout = abs(c.iloc[-1] - c.tail(20).mean()) / (recent_atr_value(df) + 1e-12)
        score = 0.0
        signals: list[str] = []
        if flag_like:
            score += 0.45 * np.sign(trend)
            signals.append("flag/pennant continuation compression")
        if triangle_like and abs(trend) > 0.2:
            score += 0.25 * np.sign(trend)
            signals.append("triangle compression aligned with prior trend")
        if breakout > 1.2:
            score += 0.25 * np.sign(c.iloc[-1] - c.tail(20).mean())
            signals.append("range breakout attempt")
        return self.result(
            score=clamp(score),
            confidence=0.4 + min(0.45, 0.12 * len(signals) + max(0.0, 1.0 - compression) * 0.2),
            direction=direction_from_score(score),
            features=_clean_features({"prior_trend": trend, "recent_slope": recent_slope, "compression_ratio": compression, "breakout_z": breakout, "flag_like": float(flag_like), "triangle_like": float(triangle_like)}),
            metadata={"signals": signals},
            signals=signals or ["no continuation pattern confirmed"],
        )


class TrendIndicatorsSkill(BaseSkill):
    definition = _definition("ta_trend_indicators")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("H1", "M15", "H4", "D1"))
        if df is None or len(df) < 35:
            return self.unavailable("need at least 35 candles for trend indicators")
        c = close(df)
        ema_fast = ema(c, 12).iloc[-1]
        ema_slow = ema(c, 26).iloc[-1]
        sma50 = sma(c, min(50, max(10, len(c) // 2))).iloc[-1]
        macd_line, macd_sig, macd_hist = macd(c)
        atr_val = recent_atr_value(df)
        ma_score = clamp((ema_fast - ema_slow) / (atr_val + 1e-12))
        sma_score = clamp((c.iloc[-1] - sma50) / (atr_val + 1e-12))
        macd_score = clamp(macd_hist.iloc[-1] / (atr_val + 1e-12) * 5.0)
        trend_slope = linear_slope_score(c, 30)
        adx_proxy = min(1.0, abs(trend_slope))
        score = clamp(0.35 * ma_score + 0.25 * sma_score + 0.25 * macd_score + 0.15 * trend_slope)
        return self.result(score=score, confidence=0.45 + 0.45 * adx_proxy, direction=direction_from_score(score), features=_clean_features({"ema_fast_slow": ma_score, "price_vs_sma": sma_score, "macd_hist_score": macd_score, "trend_slope": trend_slope, "adx_proxy": adx_proxy}), metadata={"ema_fast": float(ema_fast), "ema_slow": float(ema_slow), "sma_reference": float(sma50)}, signals=[f"trend indicator bias {direction_from_score(score)}"])


class MomentumIndicatorsSkill(BaseSkill):
    definition = _definition("ta_momentum_indicators")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("M15", "H1", "M5", "H4"))
        if df is None or len(df) < 25:
            return self.unavailable("need at least 25 candles for momentum indicators")
        c = close(df)
        rsi_v = float(rsi(c).iloc[-1])
        stoch_v = float(stochastic(df).iloc[-1])
        cci_v = float(cci(df).iloc[-1])
        roc = pct_change_score(c, 10)
        williams = -100.0 * (high(df).tail(14).max() - c.iloc[-1]) / (high(df).tail(14).max() - low(df).tail(14).min() + 1e-12)
        cmo = clamp((rsi_v - 50.0) / 50.0)
        score = clamp(0.28 * ((rsi_v - 50) / 50) + 0.22 * ((stoch_v - 50) / 50) + 0.18 * clamp(cci_v / 200) + 0.2 * roc + 0.12 * cmo)
        overextended = float(rsi_v > 70 or rsi_v < 30 or stoch_v > 85 or stoch_v < 15)
        return self.result(score=score, confidence=0.5 + 0.25 * abs(score) + 0.1 * overextended, direction=direction_from_score(score), features=_clean_features({"rsi": rsi_v / 100, "stochastic": stoch_v / 100, "cci_norm": clamp(cci_v / 200), "roc_score": roc, "williams_r_norm": (williams + 100) / 100, "cmo_score": cmo, "overextended": overextended}), metadata={"rsi": rsi_v, "stochastic": stoch_v, "cci": cci_v, "williams_r": float(williams)}, signals=["momentum bullish" if score > 0.15 else "momentum bearish" if score < -0.15 else "momentum neutral"])


class VolatilityIndicatorsSkill(BaseSkill):
    definition = _definition("ta_volatility_indicators")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("M15", "H1", "H4", "D1"))
        if df is None or len(df) < 25:
            return self.unavailable("need at least 25 candles for volatility indicators")
        c = close(df)
        atr_val = recent_atr_value(df)
        mid = sma(c, 20).iloc[-1]
        std = c.rolling(20, min_periods=8).std().iloc[-1]
        upper, lower = mid + 2 * std, mid - 2 * std
        bb_position = (c.iloc[-1] - lower) / (upper - lower + 1e-12)
        donchian_width = (high(df).tail(20).max() - low(df).tail(20).min()) / (c.iloc[-1] + 1e-12)
        atr_pct = atr_val / (c.iloc[-1] + 1e-12)
        width = (upper - lower) / (mid + 1e-12)
        expansion = clamp(width / (c.pct_change().rolling(60, min_periods=20).std().iloc[-1] + 1e-12) / 20)
        score = clamp((bb_position - 0.5) * 1.5)
        return self.result(score=score, confidence=0.45 + min(0.35, abs(bb_position - 0.5)), direction=direction_from_score(score), features=_clean_features({"atr_pct": atr_pct, "bb_position": bb_position, "bb_width": width, "donchian_width": donchian_width, "volatility_expansion": expansion}), metadata={"atr": atr_val, "bollinger_upper": float(upper), "bollinger_lower": float(lower)}, signals=["upper volatility breakout" if bb_position > 1 else "lower volatility breakdown" if bb_position < 0 else "inside volatility envelope"])


class VolumeIndicatorsSkill(BaseSkill):
    definition = _definition("ta_volume_indicators")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("M15", "H1", "M5", "H4"))
        if df is None or len(df) < 25:
            return self.unavailable("need at least 25 candles for volume indicators")
        c = close(df)
        obv_score = linear_slope_score(obv(df), 30)
        vwap_line = vwap(df)
        vwap_score = clamp((c.iloc[-1] - vwap_line.iloc[-1]) / (recent_atr_value(df) + 1e-12))
        cmf_v = float(cmf(df).iloc[-1])
        mfi_v = float(mfi(df).iloc[-1])
        vol_spike = float(volume(df).iloc[-1] / (volume(df).tail(30).mean() + 1e-12))
        spike_score = clamp((vol_spike - 1.0) / 2.0) * np.sign(c.iloc[-1] - open_(df).iloc[-1])
        score = clamp(0.3 * obv_score + 0.3 * vwap_score + 0.25 * cmf_v + 0.15 * ((mfi_v - 50) / 50) + 0.1 * spike_score)
        return self.result(score=score, confidence=0.5 + min(0.35, abs(score) * 0.35 + max(0, vol_spike - 1) * 0.08), direction=direction_from_score(score), features=_clean_features({"obv_slope": obv_score, "price_vs_vwap": vwap_score, "cmf": cmf_v, "mfi": mfi_v / 100, "volume_spike_ratio": vol_spike, "volume_spike_score": spike_score}), metadata={"vwap": float(vwap_line.iloc[-1]), "volume_spike_ratio": vol_spike}, signals=["volume confirms buying" if score > 0.15 else "volume confirms selling" if score < -0.15 else "volume neutral"])


class SupportResistanceSkill(BaseSkill):
    definition = _definition("ta_support_resistance")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("H1", "H4", "D1", "M15"))
        if df is None or len(df) < 30:
            return self.unavailable("need at least 30 candles for support/resistance")
        price = latest_price(context, df)
        atr_val = max(recent_atr_value(df), price * 0.001, 1e-12)
        highs = high(df).tail(80)
        lows = low(df).tail(80)
        support = float(lows.quantile(0.1))
        resistance = float(highs.quantile(0.9))
        pivot = float((high(df).iloc[-2] + low(df).iloc[-2] + close(df).iloc[-2]) / 3.0)
        dist_support = (price - support) / atr_val
        dist_resistance = (resistance - price) / atr_val
        near_support = dist_support >= 0 and dist_support < 1.2
        near_resistance = dist_resistance >= 0 and dist_resistance < 1.2
        breakout_up = price > resistance + 0.25 * atr_val
        breakout_down = price < support - 0.25 * atr_val
        score = 0.0
        if near_support:
            score += 0.35
        if near_resistance:
            score -= 0.35
        if breakout_up:
            score += 0.55
        if breakout_down:
            score -= 0.55
        fib_high, fib_low = float(highs.max()), float(lows.min())
        fib_618 = fib_high - 0.618 * (fib_high - fib_low)
        return self.result(score=clamp(score), confidence=0.5 + min(0.35, abs(score) * 0.4), direction=direction_from_score(score), features=_clean_features({"dist_support_atr": dist_support, "dist_resistance_atr": dist_resistance, "near_support": float(near_support), "near_resistance": float(near_resistance), "breakout_up": float(breakout_up), "breakout_down": float(breakout_down), "pivot_distance_atr": (price - pivot) / atr_val, "fib_618_distance_atr": (price - fib_618) / atr_val}), metadata={"support": support, "resistance": resistance, "pivot": pivot, "fib_618": fib_618}, signals=["near support" if near_support else "near resistance" if near_resistance else "support/resistance neutral"])


class PriceActionSkill(BaseSkill):
    definition = _definition("ta_price_action")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("H1", "M15", "H4", "D1"))
        if df is None or len(df) < 35:
            return self.unavailable("need at least 35 candles for price action")
        highs, lows = swing_highs_lows(df, window=2)
        score = linear_slope_score(close(df), 30)
        structure = "ranging"
        bos = choch = 0.0
        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1][1] > highs[-2][1]
            hl = lows[-1][1] > lows[-2][1]
            lh = highs[-1][1] < highs[-2][1]
            ll = lows[-1][1] < lows[-2][1]
            if hh and hl:
                structure = "uptrend"
                score = max(score, 0.55)
            elif lh and ll:
                structure = "downtrend"
                score = min(score, -0.55)
            price = latest_price(context, df)
            bos = float(price > highs[-1][1] or price < lows[-1][1])
            choch = float((structure == "uptrend" and price < lows[-1][1]) or (structure == "downtrend" and price > highs[-1][1]))
        return self.result(score=clamp(score), confidence=0.55 + 0.2 * abs(score) + 0.1 * bos, direction=direction_from_score(score), features=_clean_features({"structure_score": score, "bos": bos, "choch": choch, "swing_highs": len(highs), "swing_lows": len(lows)}), metadata={"market_structure": structure}, signals=[f"market structure: {structure}", "BOS" if bos else "no BOS", "CHoCH" if choch else "no CHoCH"])


class SmartMoneyConceptsSkill(BaseSkill):
    definition = _definition("ta_smart_money_concepts")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("M15", "H1", "M5", "H4"))
        if df is None or len(df) < 25:
            return self.unavailable("need at least 25 candles for SMC")
        h, l, o, c = high(df), low(df), open_(df), close(df)
        atr_val = recent_atr_value(df)
        bull_fvg = bool(l.iloc[-1] > h.iloc[-3])
        bear_fvg = bool(h.iloc[-1] < l.iloc[-3])
        sweep_sellside = bool(l.iloc[-1] < l.iloc[-20:-1].min() and c.iloc[-1] > l.iloc[-20:-1].min())
        sweep_buyside = bool(h.iloc[-1] > h.iloc[-20:-1].max() and c.iloc[-1] < h.iloc[-20:-1].max())
        impulse = (c - o).abs() / (atr_val + 1e-12)
        last_impulse = float(impulse.iloc[-1])
        bullish_ob = bool(c.iloc[-1] > o.iloc[-1] and c.iloc[-2] < o.iloc[-2] and last_impulse > 0.8)
        bearish_ob = bool(c.iloc[-1] < o.iloc[-1] and c.iloc[-2] > o.iloc[-2] and last_impulse > 0.8)
        rng_hi, rng_lo = float(h.tail(50).max()), float(l.tail(50).min())
        premium_discount = (c.iloc[-1] - rng_lo) / (rng_hi - rng_lo + 1e-12)
        score = 0.0
        signals: list[str] = []
        for cond, delta, label in [
            (bull_fvg, 0.3, "bullish FVG"),
            (bear_fvg, -0.3, "bearish FVG"),
            (sweep_sellside, 0.45, "sell-side liquidity sweep"),
            (sweep_buyside, -0.45, "buy-side liquidity sweep"),
            (bullish_ob, 0.35, "bullish order block impulse"),
            (bearish_ob, -0.35, "bearish order block impulse"),
        ]:
            if cond:
                score += delta
                signals.append(label)
        if premium_discount < 0.35:
            score += 0.1
        elif premium_discount > 0.65:
            score -= 0.1
        return self.result(score=clamp(score), confidence=0.45 + min(0.45, 0.12 * len(signals)), direction=direction_from_score(score), features=_clean_features({"bull_fvg": float(bull_fvg), "bear_fvg": float(bear_fvg), "sellside_sweep": float(sweep_sellside), "buyside_sweep": float(sweep_buyside), "bullish_order_block": float(bullish_ob), "bearish_order_block": float(bearish_ob), "premium_discount": premium_discount, "last_impulse_atr": last_impulse}), metadata={"signals": signals}, signals=signals or ["no fresh SMC footprint"])


class DivergenceAnalysisSkill(BaseSkill):
    definition = _definition("ta_divergence_analysis")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("M15", "H1", "H4", "M5"))
        if df is None or len(df) < 45:
            return self.unavailable("need at least 45 candles for divergence")
        c = close(df)
        r = rsi(c).fillna(50.0)
        o = obv(df).fillna(0.0)
        lookback = min(25, len(df) - 2)
        price_delta = float(c.iloc[-1] - c.iloc[-lookback])
        rsi_delta = float(r.iloc[-1] - r.iloc[-lookback])
        obv_delta = float(o.iloc[-1] - o.iloc[-lookback])
        regular_bull = price_delta < 0 and rsi_delta > 0
        regular_bear = price_delta > 0 and rsi_delta < 0
        hidden_bull = price_delta > 0 and rsi_delta < 0 and linear_slope_score(c, 40) > 0
        hidden_bear = price_delta < 0 and rsi_delta > 0 and linear_slope_score(c, 40) < 0
        score = 0.0
        signals: list[str] = []
        if regular_bull:
            score += 0.55
            signals.append("regular bullish RSI divergence")
        if regular_bear:
            score -= 0.55
            signals.append("regular bearish RSI divergence")
        if hidden_bull:
            score += 0.25
            signals.append("hidden bullish continuation divergence")
        if hidden_bear:
            score -= 0.25
            signals.append("hidden bearish continuation divergence")
        obv_confirm = float(np.sign(price_delta) == np.sign(obv_delta) and obv_delta != 0)
        return self.result(score=clamp(score), confidence=0.45 + min(0.4, 0.2 * len(signals)) + 0.05 * obv_confirm, direction=direction_from_score(score), features=_clean_features({"price_delta_norm": clamp(price_delta / (recent_atr_value(df) + 1e-12)), "rsi_delta_norm": clamp(rsi_delta / 50), "obv_delta_score": clamp(obv_delta / (abs(o).tail(lookback).mean() + 1e-12)), "regular_bullish": float(regular_bull), "regular_bearish": float(regular_bear), "hidden_bullish": float(hidden_bull), "hidden_bearish": float(hidden_bear), "obv_confirm": obv_confirm}), metadata={"signals": signals}, signals=signals or ["no divergence"])


class IntermarketAnalysisSkill(BaseSkill):
    definition = _definition("ta_intermarket_analysis")

    def run(self, context: Dict[str, Any]):
        correlations = context.get("correlations") or {}
        intermarket_buffers = context.get("intermarket_buffers") or {}
        df = get_df(context, preferred=("H1", "D1", "H4"))
        if not correlations and not intermarket_buffers:
            return self.result(score=0.0, confidence=0.15, features={"intermarket_available": 0.0}, metadata={"available": False, "reason": "no intermarket inputs supplied"}, signals=["intermarket data unavailable"])
        score = 0.0
        features: Dict[str, float] = {"intermarket_available": 1.0}
        if isinstance(correlations, dict):
            for key, value in correlations.items():
                try:
                    features[f"corr_{key}"] = float(value)
                    score += float(value) * 0.1
                except Exception:
                    continue
        if df is not None:
            base_returns = close(df).pct_change().tail(80)
            for symbol, other in intermarket_buffers.items():
                if is_ohlcv(other):
                    corr = float(base_returns.corr(close(other).pct_change().tail(len(base_returns))))
                    features[f"corr_{symbol}"] = np.nan_to_num(corr)
                    score += np.nan_to_num(corr) * 0.1
        return self.result(score=clamp(score), confidence=0.45, direction=direction_from_score(score), features=_clean_features(features), metadata={"correlations": features}, signals=["intermarket correlation assessed"])


class PositionSizingSkill(BaseSkill):
    definition = _definition("ta_position_sizing")

    def run(self, context: Dict[str, Any]):
        account_equity = float(context.get("account_equity") or getattr(getattr(context.get("snapshot"), "account", None), "equity", 0.0) or 0.0)
        risk_pct = float(context.get("risk_pct", 0.01))
        stop_distance = float(context.get("stop_distance") or context.get("sl_distance") or 0.0)
        df = get_df(context)
        price = latest_price(context, df)
        if stop_distance <= 0 and df is not None:
            stop_distance = max(recent_atr_value(df) * 1.5, price * 0.001)
        risk_amount = account_equity * risk_pct if account_equity > 0 else 0.0
        units = risk_amount / stop_distance if stop_distance > 0 else 0.0
        score = 0.5 if risk_pct <= 0.02 and stop_distance > 0 else -0.3
        return self.result(score=score, confidence=0.75 if stop_distance > 0 else 0.25, features=_clean_features({"account_equity": account_equity, "risk_pct": risk_pct, "risk_amount": risk_amount, "stop_distance": stop_distance, "units_raw": units}), metadata={"suggested_units": units, "risk_amount": risk_amount}, signals=[f"risk {risk_pct:.2%}, raw units {units:.2f}"])


class StopLossSkill(BaseSkill):
    definition = _definition("ta_stop_loss")

    def run(self, context: Dict[str, Any]):
        df = get_df(context)
        if df is None or len(df) < 20:
            return self.unavailable("need OHLC data for stop-loss planning")
        price = latest_price(context, df)
        action = str((context.get("decision") or {}).get("action") or context.get("action") or "HOLD").upper()
        atr_val = recent_atr_value(df)
        support = float(low(df).tail(30).min())
        resistance = float(high(df).tail(30).max())
        if action == "BUY":
            structure_stop = support - 0.2 * atr_val
            atr_stop = price - 1.5 * atr_val
            stop = min(atr_stop, structure_stop)
        elif action == "SELL":
            structure_stop = resistance + 0.2 * atr_val
            atr_stop = price + 1.5 * atr_val
            stop = max(atr_stop, structure_stop)
        else:
            stop = price
        distance = abs(price - stop)
        valid = distance > 0 and action in {"BUY", "SELL"}
        return self.result(score=0.4 if valid else 0.0, confidence=0.75 if valid else 0.25, features=_clean_features({"atr_stop_distance": 1.5 * atr_val, "structure_stop_distance": distance, "stop_distance": distance, "stop_valid": float(valid)}), metadata={"stop_loss": stop, "atr": atr_val, "support": support, "resistance": resistance}, signals=[f"{action} stop={stop:.6f}" if valid else "no active stop because no trade direction"])


class RiskRewardSkill(BaseSkill):
    definition = _definition("ta_risk_reward")

    def run(self, context: Dict[str, Any]):
        entry = float(context.get("entry") or context.get("price") or 0.0)
        stop = context.get("stop_loss") or context.get("sl")
        target = context.get("take_profit") or context.get("tp")
        df = get_df(context)
        if entry <= 0:
            entry = latest_price(context, df)
        if (stop is None or target is None) and df is not None:
            atr_val = recent_atr_value(df)
            action = str((context.get("decision") or {}).get("action") or context.get("action") or "BUY").upper()
            if action == "SELL":
                stop = entry + 1.5 * atr_val
                target = entry - 3.0 * atr_val
            else:
                stop = entry - 1.5 * atr_val
                target = entry + 3.0 * atr_val
        if entry <= 0 or stop is None or target is None:
            return self.unavailable("entry, stop and target are required")
        risk = abs(entry - float(stop))
        reward = abs(float(target) - entry)
        rrr = reward / (risk + 1e-12)
        score = clamp((rrr - 1.0) / 2.0)
        return self.result(score=score, confidence=0.85 if rrr >= 1.5 else 0.55, features=_clean_features({"risk": risk, "reward": reward, "rrr": rrr, "rrr_ok": float(rrr >= 2.0)}), metadata={"entry": entry, "stop_loss": float(stop), "take_profit": float(target)}, signals=[f"R:R={rrr:.2f}"])


class TradingPlanSkill(BaseSkill):
    definition = _definition("ta_trading_plan")

    def run(self, context: Dict[str, Any]):
        df = get_df(context)
        if df is None:
            return self.unavailable("need market data for a trading plan")
        price = latest_price(context, df)
        trend_score = linear_slope_score(close(df), 30)
        atr_val = recent_atr_value(df)
        action = (context.get("decision") or {}).get("action")
        if not action or action == "HOLD":
            action = "BUY" if trend_score > 0.25 else "SELL" if trend_score < -0.25 else "HOLD"
        if action == "BUY":
            stop, target = price - 1.5 * atr_val, price + 3.0 * atr_val
            entry_type = "pullback or breakout confirmation"
        elif action == "SELL":
            stop, target = price + 1.5 * atr_val, price - 3.0 * atr_val
            entry_type = "pullback or breakdown confirmation"
        else:
            stop = target = price
            entry_type = "wait"
        score = 0.0 if action == "HOLD" else (1.0 if action == "BUY" else -1.0) * min(0.6, abs(trend_score))
        return self.result(score=score, confidence=0.65 if action != "HOLD" else 0.4, direction=direction_from_score(score), features=_clean_features({"plan_trend_score": trend_score, "plan_atr": atr_val, "plan_rrr": 2.0 if action != "HOLD" else 0.0}), metadata={"action": action, "entry": price, "entry_type": entry_type, "stop_loss": stop, "take_profit": target, "risk_reward": 2.0 if action != "HOLD" else 0.0}, signals=[f"plan: {action} via {entry_type}"])


class TradeEntrySkill(BaseSkill):
    definition = _definition("ta_trade_entry")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("M5", "M15", "H1"))
        if df is None or len(df) < 25:
            return self.unavailable("need OHLC data for entry timing")
        price = latest_price(context, df)
        atr_val = recent_atr_value(df)
        recent_high = float(high(df).tail(20).max())
        recent_low = float(low(df).tail(20).min())
        breakout_buy = price > recent_high - 0.15 * atr_val
        breakout_sell = price < recent_low + 0.15 * atr_val
        ema20 = float(ema(close(df), 20).iloc[-1])
        pullback_buy = price >= ema20 and abs(price - ema20) <= 0.6 * atr_val and linear_slope_score(close(df), 30) > 0.2
        pullback_sell = price <= ema20 and abs(price - ema20) <= 0.6 * atr_val and linear_slope_score(close(df), 30) < -0.2
        score = (0.45 if breakout_buy else 0.0) - (0.45 if breakout_sell else 0.0) + (0.25 if pullback_buy else 0.0) - (0.25 if pullback_sell else 0.0)
        signals = []
        if breakout_buy: signals.append("breakout buy trigger")
        if breakout_sell: signals.append("breakout sell trigger")
        if pullback_buy: signals.append("bullish EMA pullback")
        if pullback_sell: signals.append("bearish EMA pullback")
        return self.result(score=clamp(score), confidence=0.45 + min(0.4, 0.15 * len(signals)), direction=direction_from_score(score), features=_clean_features({"breakout_buy": float(breakout_buy), "breakout_sell": float(breakout_sell), "pullback_buy": float(pullback_buy), "pullback_sell": float(pullback_sell), "dist_ema20_atr": (price - ema20) / (atr_val + 1e-12)}), metadata={"recent_high": recent_high, "recent_low": recent_low, "ema20": ema20}, signals=signals or ["no precision entry trigger"])


class TradeManagementSkill(BaseSkill):
    definition = _definition("ta_trade_management")

    def run(self, context: Dict[str, Any]):
        positions = getattr(context.get("snapshot"), "positions", None) or context.get("positions") or []
        decision = context.get("decision") or {}
        management = {"move_to_breakeven": False, "scale_out": False, "trail_stop": False, "reentry_allowed": False}
        score = 0.0
        if positions:
            management["trail_stop"] = True
            score += 0.2
        if float(decision.get("confidence", 0.0)) > 0.75:
            management["reentry_allowed"] = True
            score += 0.1
        return self.result(score=score, confidence=0.55, features=_clean_features({k: float(v) for k, v in management.items()} | {"open_positions": len(positions)}), metadata=management, signals=["trade management rules prepared"])


class TradeExitSkill(BaseSkill):
    definition = _definition("ta_trade_exit")

    def run(self, context: Dict[str, Any]):
        df = get_df(context)
        if df is None or len(df) < 25:
            return self.unavailable("need OHLC data for exits")
        trend = linear_slope_score(close(df), 20)
        volatility = recent_atr_value(df) / (latest_price(context, df) + 1e-12)
        time_exit = context.get("hour") in {20, 21, 22, 23}
        invalidation = abs(trend) < 0.08
        score = -0.2 if invalidation or time_exit else 0.1 * np.sign(trend)
        return self.result(score=clamp(score), confidence=0.55 + 0.15 * float(time_exit or invalidation), direction=direction_from_score(score), features=_clean_features({"exit_trend": trend, "exit_volatility": volatility, "time_exit": float(time_exit), "invalidation": float(invalidation)}), metadata={"time_exit": time_exit, "invalidation": invalidation}, signals=["exit or reduce risk" if time_exit or invalidation else "hold/trail according to plan"])


class VolumeProfileSkill(BaseSkill):
    definition = _definition("inst_volume_profile")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("M15", "H1", "M5", "H4"))
        if df is None or len(df) < 30:
            return self.unavailable("need OHLCV data for volume profile")
        tp = typical_price(df).tail(150)
        vol = volume(df).tail(len(tp))
        bins = min(32, max(8, int(np.sqrt(len(tp)))))
        hist, edges = np.histogram(tp.to_numpy(), bins=bins, weights=vol.to_numpy())
        if hist.sum() <= 0:
            hist, edges = np.histogram(tp.to_numpy(), bins=bins)
        idx = int(np.argmax(hist))
        poc = float((edges[idx] + edges[idx + 1]) / 2.0)
        order = np.argsort(hist)[::-1]
        total = hist.sum()
        selected = []
        acc = 0.0
        for i in order:
            selected.append(i)
            acc += hist[i]
            if acc >= 0.7 * total:
                break
        vah = float(edges[max(selected) + 1])
        val = float(edges[min(selected)])
        price = latest_price(context, df)
        atr_val = recent_atr_value(df)
        score = clamp((price - poc) / (atr_val + 1e-12) * 0.4)
        near_val = abs(price - val) <= atr_val
        near_vah = abs(price - vah) <= atr_val
        if near_val: score += 0.15
        if near_vah: score -= 0.15
        return self.result(score=clamp(score), confidence=0.65, direction=direction_from_score(score), features=_clean_features({"price_vs_poc_atr": (price - poc) / (atr_val + 1e-12), "near_vah": float(near_vah), "near_val": float(near_val), "profile_balance": float(np.std(hist) / (np.mean(hist) + 1e-12))}), metadata={"poc": poc, "vah": vah, "val": val, "bins": bins}, signals=["above POC" if price > poc else "below POC", "near VAL" if near_val else "near VAH" if near_vah else "inside profile"])


class OrderFlowSkill(BaseSkill):
    definition = _definition("inst_order_flow")

    def run(self, context: Dict[str, Any]):
        order_book = context.get("order_book") or {}
        bid_volume = context.get("bid_volume")
        ask_volume = context.get("ask_volume")
        features: Dict[str, float] = {}
        signals: list[str] = []
        if order_book:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            bid_size = sum(float(row[1]) for row in bids[:10]) if bids else 0.0
            ask_size = sum(float(row[1]) for row in asks[:10]) if asks else 0.0
            imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-12)
            features["book_imbalance"] = imbalance
            signals.append("order book imbalance available")
        elif bid_volume is not None and ask_volume is not None:
            bid_size = float(bid_volume)
            ask_size = float(ask_volume)
            imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-12)
            features["book_imbalance"] = imbalance
            signals.append("bid/ask volume imbalance available")
        else:
            df = get_df(context, preferred=("M1", "M5", "M15"))
            if df is None or len(df) < 10:
                return self.result(score=0.0, confidence=0.15, features={"order_flow_available": 0.0}, metadata={"available": False}, signals=["order flow data unavailable"])
            candle_delta = np.sign(close(df) - open_(df)) * volume(df)
            imbalance = float(candle_delta.tail(20).sum() / (volume(df).tail(20).sum() + 1e-12))
            features["candle_delta_proxy"] = imbalance
            signals.append("candle delta proxy used")
        score = clamp(imbalance)
        features["order_flow_available"] = 1.0
        return self.result(score=score, confidence=0.55 + 0.25 * abs(score), direction=direction_from_score(score), features=_clean_features(features), metadata={"imbalance": score}, signals=signals)


class InstitutionalSentimentSkill(BaseSkill):
    definition = _definition("inst_sentiment_analysis")

    def run(self, context: Dict[str, Any]):
        sentiment_score = float(context.get("sentiment_score", 0.0) or 0.0)
        vix = context.get("vix")
        put_call = context.get("put_call_ratio")
        news_blocked = bool(context.get("news_blocked", False))
        if vix is not None:
            sentiment_score -= clamp((float(vix) - 20.0) / 40.0) * 0.3
        if put_call is not None:
            sentiment_score -= clamp((float(put_call) - 1.0) / 1.0) * 0.2
        if news_blocked:
            sentiment_score *= 0.3
        return self.result(score=clamp(sentiment_score), confidence=0.55 if any(k in context for k in ("sentiment_score", "vix", "put_call_ratio", "news_blocked")) else 0.2, direction=direction_from_score(sentiment_score), features=_clean_features({"sentiment_score": sentiment_score, "vix_norm": 0.0 if vix is None else clamp((float(vix) - 20.0) / 40.0), "put_call_norm": 0.0 if put_call is None else clamp((float(put_call) - 1.0)), "news_blocked": float(news_blocked)}), metadata={"vix": vix, "put_call_ratio": put_call, "news_blocked": news_blocked}, signals=["sentiment risk assessed"])


class ExecutionQualitySkill(BaseSkill):
    definition = _definition("inst_execution_quality")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("M1", "M5", "M15"))
        price = latest_price(context, df)
        spread = float(context.get("spread") or getattr(getattr(context.get("snapshot"), "tick", None), "spread", 0.0) or 0.0)
        atr_val = recent_atr_value(df) if df is not None else 0.0
        spread_to_atr = spread / (atr_val + 1e-12) if atr_val > 0 else spread / max(price, 1e-12)
        quality = clamp(1.0 - spread_to_atr * 5.0, 0.0, 1.0)
        score = 0.0 if quality > 0.5 else -0.25
        return self.result(score=score, confidence=0.75, features=_clean_features({"spread": spread, "spread_to_atr": spread_to_atr, "execution_quality": quality}), metadata={"quality": quality, "avoid_execution": quality < 0.35}, signals=["execution quality OK" if quality >= 0.5 else "execution quality poor"])


class BacktestingMetricsSkill(BaseSkill):
    definition = _definition("ta_recordkeeping_metrics")

    def run(self, context: Dict[str, Any]):
        pnls = context.get("pnls") or context.get("trade_pnls") or []
        if not pnls:
            return self.result(score=0.0, confidence=0.2, features={"metrics_available": 0.0}, metadata={"available": False}, signals=["no performance history supplied"])
        arr = np.asarray(pnls, dtype=float)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        win_rate = len(wins) / len(arr)
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = abs(float(losses.mean())) if len(losses) else 0.0
        profit_factor = float(wins.sum() / (abs(losses.sum()) + 1e-12)) if len(losses) else float("inf")
        equity = arr.cumsum()
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.min(equity - peak)) if len(equity) else 0.0
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        score = clamp(expectancy / (avg_win + avg_loss + 1e-12))
        return self.result(score=score, confidence=0.75, features=_clean_features({"metrics_available": 1.0, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss, "profit_factor": min(profit_factor, 10.0), "max_drawdown_abs": max_dd, "expectancy": expectancy}), metadata={"trades": len(arr)}, signals=[f"expectancy={expectancy:.2f}, profit_factor={profit_factor:.2f}"])


class GapAnalysisSkill(BaseSkill):
    definition = _definition("ta_gap_analysis")

    def run(self, context: Dict[str, Any]):
        df = get_df(context, preferred=("D1", "H1", "M15"))
        if df is None or len(df) < 10:
            return self.unavailable("need candles for gap analysis")
        prev_close = close(df).shift(1)
        gaps = open_(df) - prev_close
        atr_val = recent_atr_value(df)
        gap = float(gaps.iloc[-1])
        gap_atr = gap / (atr_val + 1e-12)
        gap_type = "none"
        if abs(gap_atr) > 1.2:
            gap_type = "breakaway_or_exhaustion"
        elif abs(gap_atr) > 0.4:
            gap_type = "common_or_runaway"
        filled = bool((low(df).iloc[-1] <= prev_close.iloc[-1] <= high(df).iloc[-1])) if not np.isnan(prev_close.iloc[-1]) else False
        score = clamp(np.sign(gap) * min(abs(gap_atr), 1.0) * (0.5 if not filled else -0.25))
        return self.result(score=score, confidence=0.45 + min(0.35, abs(gap_atr) * 0.1), direction=direction_from_score(score), features=_clean_features({"gap_atr": gap_atr, "gap_filled": float(filled), "gap_present": float(abs(gap_atr) > 0.25)}), metadata={"gap": gap, "gap_type": gap_type, "filled": filled}, signals=[f"{gap_type} gap", "filled" if filled else "unfilled"])


EXECUTABLE_SKILLS = [
    TimeframeAnalysisSkill(),
    CandlestickPatternSkill(),
    ReversalPatternSkill(),
    ContinuationPatternSkill(),
    TrendIndicatorsSkill(),
    MomentumIndicatorsSkill(),
    VolatilityIndicatorsSkill(),
    VolumeIndicatorsSkill(),
    SupportResistanceSkill(),
    PriceActionSkill(),
    SmartMoneyConceptsSkill(),
    DivergenceAnalysisSkill(),
    IntermarketAnalysisSkill(),
    PositionSizingSkill(),
    StopLossSkill(),
    RiskRewardSkill(),
    TradingPlanSkill(),
    TradeEntrySkill(),
    TradeManagementSkill(),
    TradeExitSkill(),
    VolumeProfileSkill(),
    OrderFlowSkill(),
    InstitutionalSentimentSkill(),
    ExecutionQualitySkill(),
    BacktestingMetricsSkill(),
    GapAnalysisSkill(),
]
