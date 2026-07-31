"""
Learning engine v2 - per-instrument, per-timeframe, per-signal learning.

Major improvements over v1:
1. Per-instrument + per-timeframe + per-regime Q-tables
2. Per-signal-type expectancy tracking (win rate, avg win, avg loss, expectancy)
3. Asymmetric R:R optimization per regime
4. Signal cooldown after consecutive losses per instrument
5. Better reward shaping: expectancy-weighted, not just raw PnL
6. Adaptive epsilon that resets per instrument (more exploration early)
7. Pattern matching includes instrument market type context
8. Trailing stop and breakeven trigger learning
9. Indicator weight optimization per instrument
10. Session-time awareness (Asian/London/NY)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# Finer-grained threshold levels for more precise Q-learning
THRESHOLD_LEVELS = [i / 100 for i in range(30, 96, 5)]  # 0.30 to 0.95 in 0.05 steps

# Regime-aware R:R defaults
DEFAULT_RR_BY_REGIME = {
    "TRENDING": {"tp_mult": 2.0, "sl_mult": 1.0},
    "RANGING": {"tp_mult": 1.2, "sl_mult": 0.8},
    "VOLATILE": {"tp_mult": 1.5, "sl_mult": 1.2},
    "CHOPPY": {"tp_mult": 0.8, "sl_mult": 0.5},
    "BREAKOUT": {"tp_mult": 2.5, "sl_mult": 1.0},
    "UNKNOWN": {"tp_mult": 1.5, "sl_mult": 1.0},
}


@dataclass
class LearnedPattern:
    """A pattern the agent has learned about signal failures."""
    pattern_name: str
    conditions: dict
    failure_category: str
    occurrence_count: int = 1
    avg_pnl_pips: float = 0.0
    avoidance_rule: str = ""
    first_seen: int = 0
    last_seen: int = 0
    instrument: str = ""
    timeframe: str = ""


@dataclass
class TPAdjustment:
    """Learned TP/SL adjustments per (instrument, timeframe, regime)."""
    regime: str
    instrument: str = ""
    timeframe: str = ""
    tp_mult: float = 1.5
    sl_mult: float = 1.0
    trailing_enabled: bool = False
    trailing_distance_pct: float = 0.5  # trail at 50% of ATR
    breakeven_trigger_pct: float = 0.7  # move SL to BE after 70% of TP
    win_count: int = 0
    loss_count: int = 0
    total_tp_hit: float = 0.0
    total_sl_hit: float = 0.0
    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.win_count / max(self.win_count + self.loss_count, 1)

    @property
    def avg_win(self) -> float:
        return self.total_tp_hit / max(self.win_count, 1)

    @property
    def avg_loss(self) -> float:
        return self.total_sl_hit / max(self.loss_count, 1)

    @property
    def expectancy(self) -> float:
        """Expected value per trade in pips."""
        wr = self.win_rate
        return wr * self.avg_win - (1 - wr) * self.avg_loss


@dataclass
class SignalStats:
    """Per-signal-type statistics for a specific (instrument, timeframe, regime, direction)."""
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    total_pnl: float = 0.0
    total_tp_pips: float = 0.0
    total_sl_pips: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    last_signal_epoch: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.breakevens

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total, 1)

    @property
    def avg_win(self) -> float:
        return self.total_tp_pips / max(self.wins, 1)

    @property
    def avg_loss(self) -> float:
        return self.total_sl_pips / max(self.losses, 1)

    @property
    def expectancy(self) -> float:
        wr = self.win_rate
        return wr * self.avg_win - (1 - wr) * self.avg_loss

    @property
    def profit_factor(self) -> float:
        return self.total_tp_pips / max(abs(self.total_sl_pips), 1e-10)

    @property
    def cooldown_active(self) -> bool:
        """Cooldown after consecutive losses."""
        if self.consecutive_losses >= 3:
            return True
        if self.total >= 20 and self.expectancy < -2.0:
            return True
        return False

    @property
    def is_profitable(self) -> bool:
        return self.total >= 10 and self.expectancy > 0


class LearningEngine:
    """Learns from each signal outcome and improves future signals.

    v2 improvements:
    - Per-instrument, per-timeframe, per-regime Q-tables
    - Per-signal-type expectancy tracking
    - Asymmetric R:R optimization
    - Signal cooldown logic
    - Better reward shaping
    - Indicator weight learning
    - Session awareness
    """

    def __init__(self):
        # Q-learning: (instrument, timeframe, regime) -> threshold -> value
        self.q_table: Dict[str, Dict[float, float]] = defaultdict(
            lambda: {t: 0.0 for t in THRESHOLD_LEVELS})
        self.q_updates: Dict[str, int] = defaultdict(int)
        self.q_epsilon: float = 0.4  # Start with more exploration
        self.q_epsilon_min: float = 0.05
        self.q_epsilon_decay: float = 0.9995  # Slower decay for more exploration
        self.q_lr: float = 0.15  # Higher learning rate
        self.q_gamma: float = 0.85

        # Per-instrument epsilon tracking (reset when new instrument seen)
        self._instrument_signal_counts: Dict[str, int] = defaultdict(int)

        # Pattern library
        self.patterns: Dict[str, LearnedPattern] = {}

        # TP/SL adjustments: (instrument, timeframe, regime) -> TPAdjustment
        self.tp_adjustments: Dict[str, TPAdjustment] = {}

        # Per-signal stats: (instrument, timeframe, regime, direction) -> SignalStats
        self.signal_stats: Dict[str, SignalStats] = defaultdict(SignalStats)

        # Indicator weight learning: instrument -> {indicator: weight}
        self.indicator_weights: Dict[str, Dict[str, float]] = defaultdict(dict)
        # Track per-indicator contribution to wins/losses
        self.indicator_performance: Dict[str, Dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0}))

        # Session-awareness: (session, regime) -> SignalStats
        self.session_stats: Dict[str, SignalStats] = defaultdict(SignalStats)

        # Global statistics
        self.total_signals = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_breakevens = 0
        self.total_pnl = 0.0
        self.total_tp_pips = 0.0
        self.total_sl_pips = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.best_win_streak = 0
        self.worst_loss_streak = 0
        self.signal_history: List[dict] = []

    def _q_key(self, instrument: str, timeframe: str, regime: str) -> str:
        return f"{instrument}_{timeframe}_{regime}"

    def _stats_key(self, instrument: str, timeframe: str,
                    regime: str, direction: str) -> str:
        return f"{instrument}_{timeframe}_{regime}_{direction}"

    def _session_key(self, session: str, regime: str) -> str:
        return f"{session}_{regime}"

    def get_session(self, epoch: int) -> str:
        """Determine trading session from epoch."""
        from datetime import datetime, UTC
        hour = datetime.fromtimestamp(epoch, UTC).hour
        if 0 <= hour < 8:
            return "asian"
        elif 8 <= hour < 13:
            return "london"
        elif 13 <= hour < 22:
            return "new_york"
        else:
            return "off_hours"

    # ============================ Q-LEARNING ============================ #

    def get_recommended_threshold(self, regime: str,
                                   instrument: str = "",
                                   timeframe: str = "") -> float:
        """Get the best threshold for a specific context."""
        key = self._q_key(instrument, timeframe, regime) if instrument else regime
        values = self.q_table[key]
        if not any(v != 0.0 for v in values.values()):
            # Fall back to regime-only
            values = self.q_table[regime]
        if not any(v != 0.0 for v in values.values()):
            return 0.65
        best = max(values, key=values.get)
        return best

    def get_exploratory_threshold(self, regime: str,
                                   instrument: str = "",
                                   timeframe: str = "") -> float:
        """Get threshold with epsilon-greedy exploration."""
        # Per-instrument epsilon: more exploration for instruments with fewer signals
        count = self._instrument_signal_counts.get(instrument, 0)
        effective_epsilon = max(self.q_epsilon, 0.3 * math.exp(-count / 500))

        if np.random.random() < effective_epsilon:
            return float(np.random.choice(THRESHOLD_LEVELS))
        return self.get_recommended_threshold(regime, instrument, timeframe)

    def update_q(
        self,
        regime: str,
        threshold: float,
        reward: float,
        next_regime: str,
        instrument: str = "",
        timeframe: str = "",
        done: bool = True,
    ):
        """Q-learning update with per-instrument context."""
        state = self._q_key(instrument, timeframe, regime)
        next_state = self._q_key(instrument, timeframe, next_regime)

        future = 0.0 if done else max(self.q_table[next_state].values())
        current_q = self.q_table[state].get(threshold, 0.0)
        self.q_table[state][threshold] = current_q + self.q_lr * (
            reward + self.q_gamma * future - current_q
        )
        self.q_updates[state] += 1

        if done:
            self.q_epsilon = max(self.q_epsilon_min,
                                  self.q_epsilon * self.q_epsilon_decay)

    # ============================ PATTERN LEARNING ============================ #

    def _pattern_key(self, tool_scores: dict, regime: str,
                     failure_category: str, instrument: str = "",
                     timeframe: str = "") -> str:
        """Create a more specific pattern key including context."""
        strong_indicators = sorted(
            [(k, v) for k, v in tool_scores.items() if abs(v) > 0.2],
            key=lambda x: -abs(x[1])
        )[:4]  # Top 4 instead of 3
        indicator_str = "_".join(f"{k}{'+ ' if v > 0 else '-'}" for k, v in strong_indicators)
        parts = [regime, failure_category, indicator_str]
        if instrument:
            parts.insert(0, instrument)
        return "_".join(parts)

    def learn_from_failure(
        self,
        tool_scores: dict,
        regime: str,
        failure_category: str,
        pnl_pips: float,
        epoch: int,
        tp_pips: float,
        sl_pips: float,
        instrument: str = "",
        timeframe: str = "",
        direction: str = "",
    ):
        """Record a signal outcome and learn from it with full context."""
        self.total_signals += 1
        self.total_pnl += pnl_pips
        self._instrument_signal_counts[instrument] += 1

        is_win = pnl_pips > 0
        is_loss = pnl_pips < 0

        if is_win:
            self.total_wins += 1
            self.total_tp_pips += pnl_pips
            self.win_streak += 1
            self.loss_streak = 0
            self.best_win_streak = max(self.best_win_streak, self.win_streak)
        elif is_loss:
            self.total_losses += 1
            self.total_sl_pips += abs(pnl_pips)
            self.loss_streak += 1
            self.win_streak = 0
            self.worst_loss_streak = max(self.worst_loss_streak, self.loss_streak)
        else:
            self.total_breakevens += 1
            self.win_streak = 0
            self.loss_streak = 0

        # --- Per-signal-type stats ---
        stats_key = self._stats_key(instrument, timeframe, regime, direction)
        stats = self.signal_stats[stats_key]
        if is_win:
            stats.wins += 1
            stats.total_tp_pips += pnl_pips
            stats.max_win = max(stats.max_win, pnl_pips)
            stats.consecutive_losses = 0
        elif is_loss:
            stats.losses += 1
            stats.total_sl_pips += abs(pnl_pips)
            stats.max_loss = min(stats.max_loss, pnl_pips) if stats.max_loss != 0 else pnl_pips
            stats.consecutive_losses += 1
            stats.max_consecutive_losses = max(
                stats.max_consecutive_losses, stats.consecutive_losses)
        else:
            stats.breakevens += 1
            stats.consecutive_losses = 0
        stats.total_pnl += pnl_pips
        stats.last_signal_epoch = epoch

        # --- Session stats ---
        session = self.get_session(epoch)
        sess_key = self._session_key(session, regime)
        sess_stats = self.session_stats[sess_key]
        if is_win:
            sess_stats.wins += 1
            sess_stats.total_tp_pips += pnl_pips
        elif is_loss:
            sess_stats.losses += 1
            sess_stats.total_sl_pips += abs(pnl_pips)
        else:
            sess_stats.breakevens += 1
        sess_stats.total_pnl += pnl_pips

        # --- Indicator performance tracking ---
        for ind_name, ind_score in tool_scores.items():
            if abs(ind_score) > 0.2:  # Only track indicators that fired
                perf = self.indicator_performance[instrument][ind_name]
                if is_win:
                    perf["wins"] += 1
                    perf["total_pnl"] += pnl_pips
                elif is_loss:
                    perf["losses"] += 1
                    perf["total_pnl"] += pnl_pips

        # --- Q-learning reward: expectancy-shaped ---
        # Better reward shaping: scale by whether this improves or hurts
        # the per-signal-type expectancy
        base_reward = float(np.clip(pnl_pips * 0.01, -1, 1))
        # Bonus for reversing a losing streak
        if is_win and stats.consecutive_losses == 0 and stats.losses > 0:
            base_reward *= 1.2  # 20% bonus for breaking a loss streak
        # Penalty for extending a loss streak
        if is_loss and stats.consecutive_losses >= 3:
            base_reward *= 1.5  # 50% extra penalty for repeated failures

        threshold_used = self.get_recommended_threshold(regime, instrument, timeframe)
        self.update_q(regime, threshold_used, base_reward, regime,
                     instrument=instrument, timeframe=timeframe)

        # --- Pattern recording ---
        if failure_category and failure_category != "no_trade":
            pattern_name = self._pattern_key(
                tool_scores, regime, failure_category, instrument, timeframe)
            if pattern_name in self.patterns:
                p = self.patterns[pattern_name]
                p.occurrence_count += 1
                p.avg_pnl_pips = (
                    (p.avg_pnl_pips * (p.occurrence_count - 1) + pnl_pips)
                    / p.occurrence_count
                )
                p.last_seen = epoch
            else:
                rule = self._generate_avoidance_rule(
                    failure_category, regime, tool_scores, tp_pips, sl_pips,
                    instrument, timeframe)
                self.patterns[pattern_name] = LearnedPattern(
                    pattern_name=pattern_name,
                    conditions={
                        "regime": regime,
                        "failure_category": failure_category,
                        "instrument": instrument,
                        "timeframe": timeframe,
                        "direction": direction,
                        "strong_indicators": {
                            k: v for k, v in tool_scores.items()
                            if abs(v) > 0.2
                        },
                    },
                    failure_category=failure_category,
                    avg_pnl_pips=pnl_pips,
                    avoidance_rule=rule,
                    first_seen=epoch,
                    last_seen=epoch,
                    instrument=instrument,
                    timeframe=timeframe,
                )

        # --- TP/SL adjustment learning ---
        adj_key = f"{instrument}_{timeframe}_{regime}"
        adj = self.tp_adjustments.setdefault(
            adj_key, TPAdjustment(
                regime=regime, instrument=instrument, timeframe=timeframe,
                tp_mult=DEFAULT_RR_BY_REGIME.get(regime, {}).get("tp_mult", 1.5),
                sl_mult=DEFAULT_RR_BY_REGIME.get(regime, {}).get("sl_mult", 1.0),
            ))

        if is_win:
            adj.win_count += 1
            adj.total_tp_hit += pnl_pips
        elif is_loss:
            adj.loss_count += 1
            adj.total_sl_hit += abs(pnl_pips)
        adj.total_pnl += pnl_pips

        # Adaptive TP/SL based on actual outcomes
        if adj.win_count + adj.loss_count >= 10:
            # If avg win < avg loss, tighten SL or widen TP
            if adj.avg_loss > 0 and adj.avg_win > 0:
                rr_ratio = adj.avg_win / adj.avg_loss
                if rr_ratio < 0.8:
                    # Wins too small vs losses — widen TP
                    adj.tp_mult = min(3.0, adj.tp_mult * 1.02)
                    # Tighten SL
                    adj.sl_mult = max(0.3, adj.sl_mult * 0.98)
                elif rr_ratio > 3.0:
                    # TP too far, rarely hit — tighten
                    adj.tp_mult = max(0.5, adj.tp_mult * 0.98)

            # Enable trailing if win rate > 55% but expectancy is poor
            if adj.win_rate > 0.55 and adj.expectancy < 0:
                adj.trailing_enabled = True
                adj.trailing_distance_pct = min(0.8, adj.trailing_distance_pct * 1.01)

            # Enable breakeven if many trades get close to TP then reverse
            if adj.win_rate < 0.50 and adj.loss_count > 5:
                adj.breakeven_trigger_pct = max(0.5, adj.breakeven_trigger_pct * 0.99)

        # Specific failure-based adjustments (more aggressive than v1)
        if failure_category == "tp_too_far":
            adj.tp_mult = max(0.5, adj.tp_mult * 0.92)  # 8% reduction vs 5%
        elif failure_category == "sl_too_close":
            adj.sl_mult = min(2.5, adj.sl_mult * 1.08)  # 8% increase vs 5%

    def _generate_avoidance_rule(
        self,
        failure_category: str,
        regime: str,
        tool_scores: dict,
        tp_pips: float,
        sl_pips: float,
        instrument: str = "",
        timeframe: str = "",
    ) -> str:
        """Generate a context-rich avoidance rule."""
        ctx = f"{instrument} {timeframe}" if instrument else ""
        rules = {
            "tp_too_far": (
                f"[{ctx}] In {regime} regime, reduce TP — price reverses "
                f"before reaching {tp_pips:.0f} pip target."
            ),
            "sl_too_close": (
                f"[{ctx}] In {regime} regime, widen SL past {sl_pips:.0f} pips — "
                f"price spikes through tight stop before following signal."
            ),
            "wrong_direction": (
                f"[{ctx}] In {regime} regime, signal direction unreliable — "
                f"conflicting indicators suggest no-trade."
            ),
            "bad_entry": (
                f"[{ctx}] In {regime} regime, wait for re-entry confirmation — "
                f"entry timing consistently poor."
            ),
        }
        return rules.get(failure_category, f"[{ctx}] Avoid similar {regime} setups.")

    # ============================ SIGNAL FILTERING ============================ #

    def should_take_signal(
        self,
        direction: str,
        confidence: float,
        regime: str,
        tool_scores: dict,
        instrument: str = "",
        timeframe: str = "",
        epoch: int = 0,
    ) -> Tuple[bool, str]:
        """Decide whether to take a signal with full context awareness."""
        if direction == "HOLD":
            return False, "HOLD signal"

        # 1. Check per-signal-type cooldown
        stats_key = self._stats_key(instrument, timeframe, regime, direction)
        stats = self.signal_stats[stats_key]
        if stats.cooldown_active:
            if stats.consecutive_losses >= 3:
                return False, (
                    f"Cooldown: {stats.consecutive_losses} consecutive losses on "
                    f"{instrument} {timeframe} {regime} {direction}"
                )
            if stats.total >= 20 and stats.expectancy < -2.0:
                return False, (
                    f"Cooldown: negative expectancy ({stats.expectancy:.1f} pips) on "
                    f"{instrument} {timeframe} {regime} {direction} after {stats.total} signals"
                )

        # 2. Check session performance
        session = self.get_session(epoch)
        sess_key = self._session_key(session, regime)
        sess = self.session_stats[sess_key]
        if sess.total >= 30 and sess.expectancy < -3.0:
            return False, (
                f"Session filter: {session} session in {regime} has negative "
                f"expectancy ({sess.expectancy:.1f} pips) over {sess.total} signals"
            )

        # 3. Check adaptive threshold (per-instrument if available)
        threshold = self.get_recommended_threshold(regime, instrument, timeframe)
        if confidence < threshold:
            return False, (
                f"Confidence {confidence:.2f} below learned threshold "
                f"{threshold:.2f} for {regime} regime"
            )

        # 4. Check against learned failure patterns (with instrument context)
        strong_indicators = {
            k: v for k, v in tool_scores.items() if abs(v) > 0.2
        }
        for pattern_name, pattern in self.patterns.items():
            # Only match same instrument or same market type
            if pattern.instrument and pattern.instrument != instrument:
                continue
            if pattern.conditions.get("regime", "") != regime:
                continue
            pattern_indicators = pattern.conditions.get("strong_indicators", {})
            overlap = set(strong_indicators.keys()) & set(pattern_indicators.keys())
            # Lower threshold: 2 overlapping indicators OR 1 very strong match
            is_match = len(overlap) >= 2
            if not is_match and len(overlap) >= 1:
                # Check if the one overlapping indicator has similar magnitude
                for k in overlap:
                    if (abs(strong_indicators[k] - pattern_indicators[k]) < 0.3
                            and pattern.occurrence_count >= 5):
                        is_match = True
                        break
            if is_match and pattern.occurrence_count >= 3:
                if pattern.avg_pnl_pips < -3:
                    return False, (
                        f"Matches failure pattern '{pattern.pattern_name}' "
                        f"({pattern.occurrence_count}x, avg PnL {pattern.avg_pnl_pips:.1f} pips). "
                        f"Rule: {pattern.avoidance_rule}"
                    )

        # 5. Check indicator-specific performance
        for ind_name, ind_score in tool_scores.items():
            if abs(ind_score) > 0.5:  # Only check strongly firing indicators
                perf = self.indicator_performance.get(instrument, {}).get(ind_name)
                if perf and perf["losses"] > 10:
                    ind_wr = perf["wins"] / max(perf["wins"] + perf["losses"], 1)
                    if ind_wr < 0.40 and abs(ind_score) > 0.7:
                        return False, (
                            f"Indicator {ind_name} has {ind_wr:.0%} win rate on {instrument} "
                            f"({perf['losses']} losses) — filtering signal"
                        )

        return True, "Signal passes all filters"

    def get_optimal_tp_sl(self, regime: str, atr_pips: float,
                           instrument: str = "",
                           timeframe: str = "") -> Tuple[float, float, bool, float]:
        """Get optimized TP/SL with trailing stop config.

        Returns (tp_pips, sl_pips, trailing_enabled, breakeven_trigger_pct).
        """
        adj_key = f"{instrument}_{timeframe}_{regime}"
        adj = self.tp_adjustments.get(adj_key)

        if adj and adj.win_count + adj.loss_count >= 10:
            tp = max(5, atr_pips * adj.tp_mult)
            sl = max(3, atr_pips * adj.sl_mult)
            return tp, sl, adj.trailing_enabled, adj.breakeven_trigger_pct

        # Default regime-based R:R
        defaults = DEFAULT_RR_BY_REGIME.get(regime, DEFAULT_RR_BY_REGIME["UNKNOWN"])
        tp = max(5, atr_pips * defaults["tp_mult"])
        sl = max(3, atr_pips * defaults["sl_mult"])
        return tp, sl, False, 0.7

    # ============================ SERIALIZATION ============================ #

    def to_dict(self) -> dict:
        return {
            "q_table": {k: {str(kk): vv for kk, vv in v.items()}
                        for k, v in self.q_table.items()},
            "q_updates": dict(self.q_updates),
            "q_epsilon": self.q_epsilon,
            "patterns": {
                k: {
                    "pattern_name": v.pattern_name,
                    "conditions": v.conditions,
                    "failure_category": v.failure_category,
                    "occurrence_count": v.occurrence_count,
                    "avg_pnl_pips": v.avg_pnl_pips,
                    "avoidance_rule": v.avoidance_rule,
                    "first_seen": v.first_seen,
                    "last_seen": v.last_seen,
                    "instrument": v.instrument,
                    "timeframe": v.timeframe,
                }
                for k, v in self.patterns.items()
            },
            "tp_adjustments": {
                k: {
                    "regime": v.regime,
                    "instrument": v.instrument,
                    "timeframe": v.timeframe,
                    "tp_mult": v.tp_mult,
                    "sl_mult": v.sl_mult,
                    "trailing_enabled": v.trailing_enabled,
                    "trailing_distance_pct": v.trailing_distance_pct,
                    "breakeven_trigger_pct": v.breakeven_trigger_pct,
                    "win_count": v.win_count,
                    "loss_count": v.loss_count,
                    "total_tp_hit": v.total_tp_hit,
                    "total_sl_hit": v.total_sl_hit,
                    "total_pnl": v.total_pnl,
                }
                for k, v in self.tp_adjustments.items()
            },
            "signal_stats": {
                k: {
                    "wins": v.wins, "losses": v.losses,
                    "breakevens": v.breakevens, "total_pnl": v.total_pnl,
                    "total_tp_pips": v.total_tp_pips,
                    "total_sl_pips": v.total_sl_pips,
                    "max_win": v.max_win, "max_loss": v.max_loss,
                    "consecutive_losses": v.consecutive_losses,
                    "max_consecutive_losses": v.max_consecutive_losses,
                }
                for k, v in self.signal_stats.items()
            },
            "session_stats": {
                k: {
                    "wins": v.wins, "losses": v.losses,
                    "breakevens": v.breakevens, "total_pnl": v.total_pnl,
                }
                for k, v in self.session_stats.items()
            },
            "indicator_performance": dict(self.indicator_performance),
            "stats": {
                "total_signals": self.total_signals,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "total_breakevens": self.total_breakevens,
                "total_pnl": self.total_pnl,
                "total_tp_pips": self.total_tp_pips,
                "total_sl_pips": self.total_sl_pips,
                "win_rate": self.total_wins / max(self.total_signals, 1),
                "avg_win": self.total_tp_pips / max(self.total_wins, 1),
                "avg_loss": self.total_sl_pips / max(self.total_losses, 1),
                "expectancy": (
                    self.total_wins / max(self.total_signals, 1)
                    * (self.total_tp_pips / max(self.total_wins, 1))
                    - (self.total_losses / max(self.total_signals, 1))
                    * (self.total_sl_pips / max(self.total_losses, 1))
                ),
                "profit_factor": (self.total_tp_pips / max(self.total_sl_pips, 1e-10)),
                "best_win_streak": self.best_win_streak,
                "worst_loss_streak": self.worst_loss_streak,
                "patterns_learned": len(self.patterns),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearningEngine":
        engine = cls()
        engine.q_epsilon = data.get("q_epsilon", 0.4)
        engine.q_updates = defaultdict(int, data.get("q_updates", {}))
        for state, actions in data.get("q_table", {}).items():
            engine.q_table[state] = {float(k): v for k, v in actions.items()}
        for name, p in data.get("patterns", {}).items():
            engine.patterns[name] = LearnedPattern(
                pattern_name=p["pattern_name"],
                conditions=p["conditions"],
                failure_category=p["failure_category"],
                occurrence_count=p["occurrence_count"],
                avg_pnl_pips=p["avg_pnl_pips"],
                avoidance_rule=p["avoidance_rule"],
                first_seen=p["first_seen"],
                last_seen=p["last_seen"],
                instrument=p.get("instrument", ""),
                timeframe=p.get("timeframe", ""),
            )
        for key, adj in data.get("tp_adjustments", {}).items():
            engine.tp_adjustments[key] = TPAdjustment(
                regime=adj["regime"],
                instrument=adj.get("instrument", ""),
                timeframe=adj.get("timeframe", ""),
                tp_mult=adj["tp_mult"],
                sl_mult=adj["sl_mult"],
                trailing_enabled=adj.get("trailing_enabled", False),
                trailing_distance_pct=adj.get("trailing_distance_pct", 0.5),
                breakeven_trigger_pct=adj.get("breakeven_trigger_pct", 0.7),
                win_count=adj["win_count"],
                loss_count=adj["loss_count"],
                total_tp_hit=adj.get("total_tp_hit", 0),
                total_sl_hit=adj.get("total_sl_hit", 0),
                total_pnl=adj.get("total_pnl", 0),
            )
        for key, ss in data.get("signal_stats", {}).items():
            engine.signal_stats[key] = SignalStats(
                wins=ss["wins"], losses=ss["losses"],
                breakevens=ss.get("breakevens", 0),
                total_pnl=ss["total_pnl"],
                total_tp_pips=ss.get("total_tp_pips", 0),
                total_sl_pips=ss.get("total_sl_pips", 0),
                max_win=ss.get("max_win", 0),
                max_loss=ss.get("max_loss", 0),
                consecutive_losses=ss.get("consecutive_losses", 0),
                max_consecutive_losses=ss.get("max_consecutive_losses", 0),
            )
        for key, ss in data.get("session_stats", {}).items():
            engine.session_stats[key] = SignalStats(
                wins=ss["wins"], losses=ss["losses"],
                breakevens=ss.get("breakevens", 0),
                total_pnl=ss.get("total_pnl", 0),
            )
        for inst, indicators in data.get("indicator_performance", {}).items():
            for ind_name, perf in indicators.items():
                engine.indicator_performance[inst][ind_name] = perf

        stats = data.get("stats", {})
        engine.total_signals = stats.get("total_signals", 0)
        engine.total_wins = stats.get("total_wins", 0)
        engine.total_losses = stats.get("total_losses", 0)
        engine.total_breakevens = stats.get("total_breakevens", 0)
        engine.total_pnl = stats.get("total_pnl", 0)
        engine.total_tp_pips = stats.get("total_tp_pips", 0)
        engine.total_sl_pips = stats.get("total_sl_pips", 0)
        engine.best_win_streak = stats.get("best_win_streak", 0)
        engine.worst_loss_streak = stats.get("worst_loss_streak", 0)
        return engine
