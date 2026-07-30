"""
Trade simulator — simulates trade outcomes from historical candle data.

For each signal, simulates entering a trade and holding until TP or SL
is hit (or max hold time expires). Records the outcome, PnL, and analyzes
WHY the trade failed so the learning engine can improve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TradeOutcome:
    """Result of a simulated trade."""
    outcome: str               # win, loss, breakeven
    pnl_pips: float
    pnl_percent: float
    hold_bars: int
    failure_reason: Optional[str] = None
    failure_category: Optional[str] = None
    max_adverse: float = 0.0   # max drawdown in pips
    max_favorable: float = 0.0  # max profit in pips
    exit_price: float = 0.0
    exit_epoch: int = 0


def simulate_trade(
    signal_direction: str,
    entry_price: float,
    tp_pips: float,
    sl_pips: float,
    future_candles: list[dict],
    pip_size: float = 0.0001,
    max_hold_bars: int = 60,
) -> TradeOutcome:
    """Simulate a trade from entry through future candles.

    `future_candles` are the candles AFTER the signal, in chronological order.
    The trade is held until TP, SL, or max_hold_bars is reached.
    """
    if signal_direction == "HOLD":
        return TradeOutcome(
            outcome="breakeven", pnl_pips=0, pnl_percent=0,
            hold_bars=0, exit_price=entry_price,
            failure_reason="HOLD signal, no trade taken",
            failure_category="no_trade",
        )

    pip_value = pip_size
    direction_mult = 1 if signal_direction == "BUY" else -1

    tp_price = entry_price + direction_mult * tp_pips * pip_value
    sl_price = entry_price - direction_mult * sl_pips * pip_value

    max_adverse = 0.0
    max_favorable = 0.0
    hold_bars = 0
    exit_price = entry_price
    exit_epoch = 0

    for i, candle in enumerate(future_candles):
        if i >= max_hold_bars:
            # Time exit — close at current price
            exit_price = candle["close"]
            exit_epoch = candle["epoch"]
            hold_bars = i + 1
            break

        high = candle["high"]
        low = candle["low"]

        # Track max adverse and favorable excursion
        if direction_mult == 1:  # BUY
            adverse = (entry_price - low) / pip_value
            favorable = (high - entry_price) / pip_value
        else:  # SELL
            adverse = (high - entry_price) / pip_value
            favorable = (entry_price - low) / pip_value

        max_adverse = max(max_adverse, adverse)
        max_favorable = max(max_favorable, favorable)

        # Check SL hit
        if direction_mult == 1 and low <= sl_price:
            exit_price = sl_price
            exit_epoch = candle["epoch"]
            hold_bars = i + 1
            break
        if direction_mult == -1 and high >= sl_price:
            exit_price = sl_price
            exit_epoch = candle["epoch"]
            hold_bars = i + 1
            break

        # Check TP hit
        if direction_mult == 1 and high >= tp_price:
            exit_price = tp_price
            exit_epoch = candle["epoch"]
            hold_bars = i + 1
            break
        if direction_mult == -1 and low <= tp_price:
            exit_price = tp_price
            exit_epoch = candle["epoch"]
            hold_bars = i + 1
            break

        exit_price = candle["close"]
        exit_epoch = candle["epoch"]
        hold_bars = i + 1

    # Calculate PnL
    pnl_price = (exit_price - entry_price) * direction_mult
    pnl_pips = pnl_price / pip_value
    pnl_percent = (pnl_price / entry_price) * 100 if entry_price else 0

    # Determine outcome
    if pnl_pips >= tp_pips * 0.9:
        outcome = "win"
    elif pnl_pips <= -sl_pips * 0.9:
        outcome = "loss"
    elif abs(pnl_pips) < 2:
        outcome = "breakeven"
    elif pnl_pips > 0:
        outcome = "win"
    else:
        outcome = "loss"

    # Analyze failure
    failure_reason = None
    failure_category = None

    if outcome == "loss":
        if max_favorable >= tp_pips * 0.5:
            failure_reason = (
                f"Price reached {max_favorable:.1f} pips favorable but "
                f"reversed to hit SL. TP may be too far ({tp_pips:.1f} pips)."
            )
            failure_category = "tp_too_far"
        elif max_adverse >= sl_pips * 0.8 and hold_bars <= 3:
            failure_reason = (
                f"SL hit quickly ({hold_bars} bars). Entry timing was poor "
                f"or SL too close ({sl_pips:.1f} pips)."
            )
            failure_category = "sl_too_close"
        elif hold_bars >= max_hold_bars:
            failure_reason = (
                f"Trade timed out after {hold_bars} bars. "
                f"Signal direction may be wrong or market was ranging."
            )
            failure_category = "wrong_direction"
        elif max_adverse > max_favorable * 2:
            failure_reason = (
                f"Max adverse ({max_adverse:.1f}) was much larger than "
                f"max favorable ({max_favorable:.1f}). Signal was likely "
                f"against the trend."
            )
            failure_category = "wrong_direction"
        else:
            failure_reason = (
                f"Trade failed. PnL: {pnl_pips:.1f} pips. "
                f"Max favorable: {max_favorable:.1f}, "
                f"max adverse: {max_adverse:.1f}."
            )
            failure_category = "bad_entry"

    return TradeOutcome(
        outcome=outcome,
        pnl_pips=round(pnl_pips, 2),
        pnl_percent=round(pnl_percent, 4),
        hold_bars=hold_bars,
        failure_reason=failure_reason,
        failure_category=failure_category,
        max_adverse=round(max_adverse, 2),
        max_favorable=round(max_favorable, 2),
        exit_price=exit_price,
        exit_epoch=exit_epoch,
    )
