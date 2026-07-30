"""
Learning engine — learns from each signal failure and improves.

After each simulated trade, this module:
1. Records the failure pattern (market conditions + failure category)
2. Adjusts the adaptive threshold via Q-learning
3. Generates an avoidance rule so future signals in similar conditions
   are filtered or adjusted
4. Updates TP/SL recommendations based on what worked
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# Q-learning threshold levels
THRESHOLD_LEVELS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


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


@dataclass
class TPAdjustment:
    """Learned TP/SL adjustments per regime."""
    regime: str
    tp_multiplier: float = 1.5   # ATR multiplier
    sl_multiplier: float = 0.8
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0


class LearningEngine:
    """Learns from each signal failure and improves future signals.

    Maintains:
    - Q-learning table for adaptive thresholds per regime
    - Pattern library of failure conditions
    - TP/SL adjustment factors per regime
    - Running statistics for introspection
    """

    def __init__(self):
        # Q-learning: state=regime -> action=threshold level
        self.q_table: Dict[str, Dict[float, float]] = defaultdict(
            lambda: {t: 0.0 for t in THRESHOLD_LEVELS})
        self.q_updates: Dict[str, int] = defaultdict(int)
        self.q_epsilon: float = 0.3
        self.q_epsilon_min: float = 0.05
        self.q_epsilon_decay: float = 0.998
        self.q_lr: float = 0.1
        self.q_gamma: float = 0.9

        # Pattern library
        self.patterns: Dict[str, LearnedPattern] = {}

        # TP/SL adjustments
        self.tp_adjustments: Dict[str, TPAdjustment] = {}

        # Statistics
        self.total_signals = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.best_win_streak = 0
        self.worst_loss_streak = 0
        self.signal_history: List[dict] = []

    # ============================ Q-LEARNING ============================ #

    def get_recommended_threshold(self, regime: str) -> float:
        """Get the best threshold for a regime (exploit, no exploration)."""
        state = regime.upper()
        values = self.q_table[state]
        if not any(values.values()):
            return 0.65  # default
        best = max(values, key=values.get)
        return best

    def get_exploratory_threshold(self, regime: str) -> float:
        """Get threshold with epsilon-greedy exploration for training."""
        state = regime.upper()
        if np.random.random() < self.q_epsilon:
            return float(np.random.choice(THRESHOLD_LEVELS))
        return self.get_recommended_threshold(state)

    def update_q(
        self,
        regime: str,
        threshold: float,
        reward: float,
        next_regime: str,
        done: bool = True,
    ):
        """Q-learning update."""
        state = regime.upper()
        next_state = next_regime.upper()

        future = 0.0 if done else max(self.q_table[next_state].values())
        current_q = self.q_table[state][threshold]
        self.q_table[state][threshold] = current_q + self.q_lr * (
            reward + self.q_gamma * future - current_q
        )
        self.q_updates[state] += 1

        if done:
            self.q_epsilon = max(self.q_epsilon_min, self.q_epsilon * self.q_epsilon_decay)

    # ============================ PATTERN LEARNING ============================ #

    def _pattern_key(self, tool_scores: dict, regime: str,
                     failure_category: str) -> str:
        """Create a pattern key from the dominant conditions."""
        # Identify which indicators were strong
        strong_indicators = sorted(
            [(k, v) for k, v in tool_scores.items() if abs(v) > 0.3],
            key=lambda x: -abs(x[1])
        )[:3]
        indicator_str = "_".join(f"{k}{'+ ' if v > 0 else '-'}" for k, v in strong_indicators)
        return f"{regime}_{failure_category}_{indicator_str}"

    def learn_from_failure(
        self,
        tool_scores: dict,
        regime: str,
        failure_category: str,
        pnl_pips: float,
        epoch: int,
        tp_pips: float,
        sl_pips: float,
    ):
        """Record a failure pattern and learn from it."""
        self.total_signals += 1
        self.total_pnl += pnl_pips

        if pnl_pips > 0:
            self.total_wins += 1
            self.win_streak += 1
            self.loss_streak = 0
            self.best_win_streak = max(self.best_win_streak, self.win_streak)
        else:
            self.total_losses += 1
            self.loss_streak += 1
            self.win_streak = 0
            self.worst_loss_streak = max(self.worst_loss_streak, self.loss_streak)

        # Q-learning reward
        reward = float(np.clip(pnl_pips * 0.1, -1, 1))
        threshold_used = 0.65  # default during training
        self.update_q(regime, threshold_used, reward, regime)

        # Pattern recording
        if failure_category and failure_category != "no_trade":
            pattern_name = self._pattern_key(tool_scores, regime, failure_category)
            if pattern_name in self.patterns:
                p = self.patterns[pattern_name]
                p.occurrence_count += 1
                p.avg_pnl_pips = (
                    (p.avg_pnl_pips * (p.occurrence_count - 1) + pnl_pips)
                    / p.occurrence_count
                )
                p.last_seen = epoch
            else:
                # Generate avoidance rule
                rule = self._generate_avoidance_rule(
                    failure_category, regime, tool_scores, tp_pips, sl_pips)
                self.patterns[pattern_name] = LearnedPattern(
                    pattern_name=pattern_name,
                    conditions={
                        "regime": regime,
                        "failure_category": failure_category,
                        "strong_indicators": {
                            k: v for k, v in tool_scores.items()
                            if abs(v) > 0.3
                        },
                    },
                    failure_category=failure_category,
                    avg_pnl_pips=pnl_pips,
                    avoidance_rule=rule,
                    first_seen=epoch,
                    last_seen=epoch,
                )

        # TP/SL adjustment learning
        adj = self.tp_adjustments.setdefault(
            regime, TPAdjustment(regime=regime))
        if pnl_pips > 0:
            adj.win_count += 1
        else:
            adj.loss_count += 1
        adj.total_pnl += pnl_pips

        # Adjust multipliers based on outcomes
        if failure_category == "tp_too_far" and adj.loss_count > 0:
            # TP was too far — reduce it
            adj.tp_multiplier = max(0.5, adj.tp_multiplier * 0.95)
        elif failure_category == "sl_too_close" and adj.loss_count > 0:
            # SL was too close — widen it
            adj.sl_multiplier = min(2.0, adj.sl_multiplier * 1.05)
        elif failure_category == "wrong_direction" and adj.loss_count > 0:
            # Direction was wrong — this is a signal quality issue, not TP/SL
            pass

    def _generate_avoidance_rule(
        self,
        failure_category: str,
        regime: str,
        tool_scores: dict,
        tp_pips: float,
        sl_pips: float,
    ) -> str:
        """Generate a human-readable avoidance rule."""
        rules = {
            "tp_too_far": (
                f"In {regime} market, when these indicators fire, "
                f"reduce TP to {tp_pips * 0.6:.0f} pips — price reverses "
                f"before reaching full target."
            ),
            "sl_too_close": (
                f"In {regime} market, widen SL to {sl_pips * 1.5:.0f} pips — "
                f"price spikes past tight SL before going in signal direction."
            ),
            "wrong_direction": (
                f"In {regime} market, do not take signals when these "
                f"indicators conflict — the signal direction is likely wrong."
            ),
            "bad_entry": (
                f"In {regime} market, wait for better entry confirmation "
                f"when these indicators fire — entry timing was poor."
            ),
        }
        return rules.get(failure_category, f"Avoid similar setups in {regime} market.")

    # ============================ SIGNAL FILTERING ============================ #

    def should_take_signal(
        self,
        direction: str,
        confidence: float,
        regime: str,
        tool_scores: dict,
    ) -> Tuple[bool, str]:
        """Decide whether to take a signal, based on learned patterns.

        Returns (should_take, reason).
        """
        if direction == "HOLD":
            return False, "HOLD signal"

        # Check adaptive threshold
        threshold = self.get_recommended_threshold(regime)
        if confidence < threshold:
            return False, (
                f"Confidence {confidence:.2f} below learned threshold "
                f"{threshold:.2f} for {regime} regime"
            )

        # Check against learned failure patterns
        strong_indicators = {
            k: v for k, v in tool_scores.items() if abs(v) > 0.3
        }
        for pattern_name, pattern in self.patterns.items():
            if pattern.conditions.get("regime", "") != regime:
                continue
            pattern_indicators = pattern.conditions.get("strong_indicators", {})
            # Check if current strong indicators match a known failure pattern
            overlap = set(strong_indicators.keys()) & set(pattern_indicators.keys())
            if len(overlap) >= 2 and pattern.occurrence_count >= 3:
                if pattern.avg_pnl_pips < -5:
                    return False, (
                        f"Matches failure pattern '{pattern.pattern_name}' "
                        f"(occurred {pattern.occurrence_count}x, "
                        f"avg PnL {pattern.avg_pnl_pips:.1f} pips). "
                        f"Rule: {pattern.avoidance_rule}"
                    )

        return True, "Signal passes all filters"

    # ============================ SERIALIZATION ============================ #

    def to_dict(self) -> dict:
        return {
            "q_table": {k: dict(v) for k, v in self.q_table.items()},
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
                }
                for k, v in self.patterns.items()
            },
            "tp_adjustments": {
                k: {
                    "regime": v.regime,
                    "tp_multiplier": v.tp_multiplier,
                    "sl_multiplier": v.sl_multiplier,
                    "win_count": v.win_count,
                    "loss_count": v.loss_count,
                    "total_pnl": v.total_pnl,
                }
                for k, v in self.tp_adjustments.items()
            },
            "stats": {
                "total_signals": self.total_signals,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "total_pnl": self.total_pnl,
                "win_rate": self.total_wins / max(self.total_signals, 1),
                "best_win_streak": self.best_win_streak,
                "worst_loss_streak": self.worst_loss_streak,
                "patterns_learned": len(self.patterns),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearningEngine":
        engine = cls()
        engine.q_epsilon = data.get("q_epsilon", 0.3)
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
            )
        for regime, adj in data.get("tp_adjustments", {}).items():
            engine.tp_adjustments[regime] = TPAdjustment(
                regime=adj["regime"],
                tp_multiplier=adj["tp_multiplier"],
                sl_multiplier=adj["sl_multiplier"],
                win_count=adj["win_count"],
                loss_count=adj["loss_count"],
                total_pnl=adj["total_pnl"],
            )
        stats = data.get("stats", {})
        engine.total_signals = stats.get("total_signals", 0)
        engine.total_wins = stats.get("total_wins", 0)
        engine.total_losses = stats.get("total_losses", 0)
        engine.total_pnl = stats.get("total_pnl", 0)
        engine.best_win_streak = stats.get("best_win_streak", 0)
        engine.worst_loss_streak = stats.get("worst_loss_streak", 0)
        return engine
