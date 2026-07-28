"""
signal_engine.py
================
Wraps Veilcrean's 144-tool ConfluenceEngine and turns its aggregate read of a
market into a concrete trade signal: DIRECTION + ENTRY + TP + SL + confidence.

The per-instrument learning parameters (threshold, confidence multiplier,
risk-reward, SL multiple) are injected by generate_signals.py so the engine's
behaviour adapts as past signals are scored.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd

# make the repo importable no matter where this runs from
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from python_brain.confluence.confluence_engine import ConfluenceEngine   # noqa: E402
from python_brain.communication.data_parser import MarketSnapshot, TickData  # noqa: E402


class _BDF(pd.DataFrame):
    """DataFrame with a defined truthiness so the tools' `a or b` buffer
    fallback pattern works instead of raising 'ambiguous truth value'."""
    @property
    def _constructor(self):
        return _BDF

    def __bool__(self):
        return len(self.index) > 0


def _wrap(buffers: Dict[str, pd.DataFrame]) -> Dict[str, _BDF]:
    return {k: _BDF(v) for k, v in buffers.items()}


# single shared engine instance (loading 144 tools is not free)
_ENGINE: ConfluenceEngine | None = None


def _engine() -> ConfluenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ConfluenceEngine()
    return _ENGINE


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    if not np.isfinite(val) or val <= 0:
        val = float((df["high"] - df["low"]).tail(period).mean())
    return float(val)


def _robust_aggregate(tool_results) -> tuple[float, int, int, int, float]:
    """Weighted mean score ignoring NaNs. Returns (score, buys, sells,
    neutral, mean_conf)."""
    num = den = 0.0
    buys = sells = neutral = 0
    confs = []
    for r in tool_results.values():
        s, c = r.score, r.confidence
        if s is None or not np.isfinite(s):
            continue
        c = 0.5 if (c is None or not np.isfinite(c)) else float(c)
        w = max(0.1, c)
        num += s * w
        den += w
        confs.append(c)
        if s > 0.05:
            buys += 1
        elif s < -0.05:
            sells += 1
        else:
            neutral += 1
    score = num / den if den else 0.0
    return float(np.clip(score, -1, 1)), buys, sells, neutral, (
        float(np.mean(confs)) if confs else 0.5)


def _round_for(price: float, x: float) -> float:
    if price >= 1000:
        return round(x, 2)
    if price >= 100:
        return round(x, 3)
    return round(x, 5)


def analyse(name: str, buffers: Dict[str, pd.DataFrame], price: float,
            params: dict, force: str | None = None) -> dict:
    """
    Run the full tool suite and produce a trade signal dict.

    params keys (per-instrument, supplied by the learning layer):
        threshold        min |adj_score| to call a directional trade
        conf_multiplier  scales raw aggregate score
        rr               reward-to-risk ratio (TP dist / SL dist)
        k_sl             SL distance as a multiple of ATR
    """
    eng = _engine()
    tick = TickData(bid=price, ask=price * 1.00002, spread=2.0, volume=100)
    snap = MarketSnapshot(symbol=name, trigger="CANDLE",
                          timestamp=datetime.now(timezone.utc),
                          tick=tick, candles=_wrap(buffers))
    res = eng.run(snap, _wrap(buffers))
    raw_score, buys, sells, neutral, mean_conf = _robust_aggregate(res["tool_results"])

    adj_score = float(np.clip(raw_score * params.get("conf_multiplier", 1.0), -1, 1))
    threshold = params.get("threshold", 0.04)

    # ---- direction -------------------------------------------------------- #
    if force in ("BUY", "SELL"):
        direction = force
        conviction = "FORCED"
    elif adj_score >= threshold:
        direction = "BUY"
        conviction = "NORMAL"
    elif adj_score <= -threshold:
        direction = "SELL"
        conviction = "NORMAL"
    else:
        # below threshold -> follow the tilt but flag as low conviction
        direction = "BUY" if adj_score >= 0 else "SELL"
        conviction = "LOW"

    # ---- levels ----------------------------------------------------------- #
    # ATR from the fast ("M15") slot -> hourly for swing, 5-minute for scalp,
    # so scalp targets are naturally tight.
    atr = _atr(buffers.get("M15", buffers.get("H1")))
    k_sl = params.get("k_sl", 1.5)
    rr = params.get("rr", 2.0)
    sl_dist = k_sl * atr
    tp_dist = rr * sl_dist

    if direction == "BUY":
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist

    total = max(1, buys + sells + neutral)
    agree = (buys if direction == "BUY" else sells) / total
    confidence = round(float(np.clip(
        0.35 * abs(adj_score) / max(threshold, 1e-6) * 0.5
        + 0.4 * agree + 0.25 * mean_conf, 0, 1)), 3)

    return {
        "instrument": name,
        "direction": direction,
        "conviction": conviction,
        "entry": _round_for(price, price),
        "tp": _round_for(price, tp),
        "sl": _round_for(price, sl),
        "atr": _round_for(price, atr),
        "rr": round(rr, 2),
        "raw_score": round(raw_score, 4),
        "adj_score": round(adj_score, 4),
        "confidence": confidence,
        "votes": {"buy": buys, "sell": sells, "neutral": neutral},
        "tools_ok": sum(1 for r in res["tool_results"].values() if not r.errors),
        "tools_total": len(res["tool_results"]),
    }
