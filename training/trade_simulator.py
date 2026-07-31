from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TradeOutcome:
    """Result of a simulated trade."""
    outcome: str               # win, loss, breakeven
    pnl_pips: float
    pnl_percent: float
    hold_bars: int
    failure_reason: Optional[str] = None
    failure_category: Optional[str] = None
    max_adverse: float = 0.0
    max_favorable: float = 0.0
    exit_price: float = 0.0
    exit_epoch: int = 0
    trailing_activated: bool = False
    breakeven_activated: bool = False
    exit_type: str = "time"


def simulate_trade(
    signal_direction: str,
    entry_price: float,
    tp_pips: float,
    sl_pips: float,
    future_candles: list[dict],
    pip_size: float = 0.0001,
    max_hold_bars: int = 20,
    trailing_enabled: bool = False,
    trailing_distance_pct: float = 0.5,
    breakeven_trigger_pct: float = 0.7,
) -> TradeOutcome:
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

    trailing_activated = False
    breakeven_activated = False
    trail_price = sl_price

    max_adverse = 0.0
    max_favorable = 0.0
    hold_bars = 0
    exit_price = entry_price
    exit_epoch = 0
    exit_type = "time"

    for i, candle in enumerate(future_candles):
        if i >= max_hold_bars:
            exit_price = candle["close"]
            exit_epoch = candle["epoch"]
            hold_bars = i + 1
            exit_type = "time"
            break

        high = candle["high"]
        low = candle["low"]
        open_price = candle.get("open", candle["close"])
        close = candle["close"]

        if direction_mult == 1:
            adverse = (entry_price - low) / pip_value
            favorable = (high - entry_price) / pip_value
        else:
            adverse = (high - entry_price) / pip_value
            favorable = (entry_price - low) / pip_value

        max_adverse = max(max_adverse, adverse)
        max_favorable = max(max_favorable, favorable)

        check_points = [("open", open_price), ("high", high), ("low", low), ("close", close)]

        for point_name, point_price in check_points:
            if trailing_activated:
                if direction_mult == 1 and point_price <= trail_price:
                    exit_price = trail_price
                    exit_epoch = candle["epoch"]
                    hold_bars = i + 1
                    exit_type = "trailing"
                    break
                if direction_mult == -1 and point_price >= trail_price:
                    exit_price = trail_price
                    exit_epoch = candle["epoch"]
                    hold_bars = i + 1
                    exit_type = "trailing"
                    break

            if breakeven_activated:
                be_sl = entry_price - direction_mult * 0.5 * pip_value
                if direction_mult == 1 and point_price <= be_sl:
                    exit_price = entry_price
                    exit_epoch = candle["epoch"]
                    hold_bars = i + 1
                    exit_type = "breakeven_stop"
                    break
                if direction_mult == -1 and point_price >= be_sl:
                    exit_price = entry_price
                    exit_epoch = candle["epoch"]
                    hold_bars = i + 1
                    exit_type = "breakeven_stop"
                    break

            if direction_mult == 1 and point_price <= sl_price:
                exit_price = sl_price
                exit_epoch = candle["epoch"]
                hold_bars = i + 1
                exit_type = "sl"
                break
            if direction_mult == -1 and point_price >= sl_price:
                exit_price = sl_price
                exit_epoch = candle["epoch"]
                hold_bars = i + 1
                exit_type = "sl"
                break

            if direction_mult == 1 and point_price >= tp_price:
                exit_price = tp_price
                exit_epoch = candle["epoch"]
                hold_bars = i + 1
                exit_type = "tp"
                break
            if direction_mult == -1 and point_price <= tp_price:
                exit_price = tp_price
                exit_epoch = candle["epoch"]
                hold_bars = i + 1
                exit_type = "tp"
                break

            if not trailing_activated and trailing_enabled and max_favorable >= tp_pips * 0.5:
                trailing_activated = True
                trail_dist = trailing_distance_pct * (tp_pips + sl_pips) / 2
                current_profit = favorable
                trail_price = entry_price + direction_mult * (current_profit - trail_dist) * pip_value
                if direction_mult == 1:
                    trail_price = max(trail_price, entry_price + pip_value)
                else:
                    trail_price = min(trail_price, entry_price - pip_value)

            if not breakeven_activated and max_favorable >= tp_pips * breakeven_trigger_pct:
                breakeven_activated = True

            if trailing_activated:
                if direction_mult == 1:
                    new_trail = point_price - trailing_distance_pct * (tp_pips + sl_pips) / 2 * pip_value
                    if new_trail > trail_price:
                        trail_price = new_trail
                else:
                    new_trail = point_price + trailing_distance_pct * (tp_pips + sl_pips) / 2 * pip_value
                    if new_trail < trail_price:
                        trail_price = new_trail
        else:
            exit_price = close
            exit_epoch = candle["epoch"]
            hold_bars = i + 1
            continue

        break

    pnl_price = (exit_price - entry_price) * direction_mult
    pnl_pips = pnl_price / pip_value
    pnl_percent = (pnl_price / entry_price) * 100 if entry_price else 0

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

    failure_reason = None
    failure_category = None

    if outcome == "loss":
        if exit_type == "trailing" and max_favorable >= tp_pips * 0.7:
            failure_reason = f"Trailing stop exited at {pnl_pips:.1f}p after {max_favorable:.1f}p favorable. Trail too tight."
            failure_category = "trail_too_tight"
        elif exit_type == "breakeven_stop":
            failure_reason = f"BE stop hit after {max_favorable:.1f}p favorable then reversed."
            failure_category = "be_reversal"
        elif max_favorable >= tp_pips * 0.5:
            failure_reason = f"Price reached {max_favorable:.1f}p favorable but reversed. TP ({tp_pips:.1f}) too far."
            failure_category = "tp_too_far"
        elif max_adverse >= sl_pips * 0.8 and hold_bars <= 3:
            failure_reason = f"SL hit in {hold_bars} bars. Entry poor or SL ({sl_pips:.1f}) too close."
            failure_category = "sl_too_close"
        elif hold_bars >= max_hold_bars:
            failure_reason = f"Timed out after {hold_bars} bars. Direction wrong."
            failure_category = "wrong_direction"
        elif max_adverse > max_favorable * 2:
            failure_reason = f"Adverse ({max_adverse:.1f}) >> favorable ({max_favorable:.1f}). Against trend."
            failure_category = "wrong_direction"
        else:
            failure_reason = f"Failed ({exit_type}). PnL: {pnl_pips:.1f}p. Fav: {max_favorable:.1f}, Adv: {max_adverse:.1f}."
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
        trailing_activated=trailing_activated,
        breakeven_activated=breakeven_activated,
        exit_type=exit_type,
    )
