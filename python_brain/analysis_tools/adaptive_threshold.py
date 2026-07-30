"""
adaptive_threshold.py
=====================
NeuroSense Q-learning for adaptive confidence thresholds.

Uses reinforcement learning to learn what confidence threshold
produces the best trade outcomes in different market regimes.
The agent observes the current regime as its state, chooses a
threshold level as its action, and receives reward based on whether
the subsequent trades were profitable.
"""
from __future__ import annotations
import time
from typing import Dict, Optional, List
import pandas as pd

from .base_tool import BaseTool, ToolResult


# Discrete threshold levels the agent can choose from
THRESHOLD_LEVELS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Regime labels mapped to state keys
REGIME_STATES = {
    "TRENDING": "regime_trending",
    "RANGING": "regime_ranging",
    "VOLATILE": "regime_volatile",
    "CHOPPY": "regime_choppy",
    "BREAKOUT": "regime_breakout",
    "UNKNOWN": "regime_unknown",
}


class AdaptiveThresholdTool(BaseTool):
    """Learns the optimal confidence threshold per regime via Q-learning.

    Instead of a static confidence threshold, this tool uses neurosense's
    QLearner to discover which threshold works best in each market regime.
    After each trade, call record_outcome() to teach the agent.
    """
    name = "adaptive_threshold"

    def __init__(self):
        super().__init__()
        self.last_run = 0
        self.last_result: Optional[ToolResult] = None
        self._init_agent()

    def _init_agent(self):
        try:
            from neurosense.learning.reinforcement import QLearner
            self.agent = QLearner(
                actions=THRESHOLD_LEVELS,
                lr=0.1,
                gamma=0.9,
                epsilon=0.3,
                epsilon_min=0.05,
                epsilon_decay=0.99,
                seed=42,
            )
            self.enabled = True
        except Exception:
            self.agent = None
            self.enabled = False

    def get_recommended_threshold(self, regime: str) -> float:
        """Returns the Q-learning agent's recommended threshold for a regime."""
        if not self.enabled:
            return 0.65  # default fallback

        state = REGIME_STATES.get(regime, "regime_unknown")
        # Use best_action (exploit, no exploration) for recommendations
        return self.agent.best_action(state)

    def record_outcome(self, regime: str, threshold: float,
                       trade_pnl: float):
        """Teach the agent: was this threshold choice good for this regime?

        Call this after each trade closes.
        """
        if not self.enabled:
            return

        state = REGIME_STATES.get(regime, "regime_unknown")
        # Reward: positive for wins, negative for losses, scaled
        reward = max(-1.0, min(1.0, trade_pnl * 0.1))
        # Next state is the same regime (simplified)
        self.agent.learn(state, threshold, reward, state, done=True)

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        if not self.enabled:
            return ToolResult(tool_name=self.name, score=0.0, confidence=0.0)

        now = time.time()
        if self.last_result and (now - self.last_run) < 30:
            return self.last_result

        regime = ctx.get("regime", "UNKNOWN")
        current_confidence = ctx.get("confidence", 0.65)

        recommended = self.get_recommended_threshold(regime)

        # Score: how well does the current confidence compare to the
        # Q-learning recommendation? Positive if current confidence
        # exceeds the recommended threshold (meaning the setup is strong
        # enough to trade even by the adaptive standard).
        if current_confidence >= recommended:
            score = min(1.0, (current_confidence - recommended) * 5 + 0.3)
        else:
            score = -0.2  # below recommended threshold — weak setup

        confidence = 0.5 + 0.05 * self.agent.total_updates
        confidence = min(1.0, confidence)

        result = ToolResult(
            tool_name=self.name,
            score=float(score),
            confidence=float(confidence),
            features={
                "recommended_threshold": recommended,
                "current_confidence": current_confidence,
                "regime": regime,
                "agent_updates": self.agent.total_updates,
                "exploration_rate": self.agent.epsilon,
            },
            metadata={
                "reasoning": f"Q-learning recommends {recommended:.2f} threshold "
                             f"for {regime} regime (current: {current_confidence:.2f})."
            }
        )
        self.last_run = now
        self.last_result = result
        return result
