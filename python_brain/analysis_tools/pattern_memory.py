"""
pattern_memory.py
=================
NeuroSense episodic memory for market pattern recognition.

Records market setups as episodic memories and recalls similar past
setups to inform the current decision. Each recorded episode stores
the tool scores, the market regime, and the eventual trade outcome.
When a new setup appears, the tool recalls the most similar past
episodes and scores the current setup by how those past trades
performed.
"""
from __future__ import annotations
import time
import json
from typing import Dict, Optional, List
import pandas as pd

from .base_tool import BaseTool, ToolResult


class PatternMemoryTool(BaseTool):
    """Recalls similar past market setups from episodic memory.

    Uses neurosense's EpisodicMemory to store and retrieve market
    episodes. Each episode's summary encodes the key tool scores so
    that keyword-based recall can find similar setups.
    """
    name = "pattern_memory"

    def __init__(self):
        super().__init__()
        self.last_run = 0
        self.last_result: Optional[ToolResult] = None
        self._init_memory()

    def _init_memory(self):
        try:
            from neurosense.brain.memory import EpisodicMemory
            self.memory = EpisodicMemory(max_episodes=5000)
            self.enabled = True
        except Exception:
            self.memory = None
            self.enabled = False

    def record_trade(self, tool_scores: Dict[str, float],
                     regime: str, outcome: str, pnl: float):
        """Record a completed trade as an episodic memory.

        Called after a trade closes so future setups can recall it.
        """
        if not self.enabled:
            return

        # Encode the setup as a keyword-rich summary for recall
        score_keys = [k for k, v in tool_scores.items() if abs(v) > 0.2]
        summary = f"{regime} {outcome} {' '.join(score_keys)} pnl={pnl:.2f}"
        importance = min(1.0, 0.3 + abs(pnl) * 0.1)

        self.memory.record(
            kind="trade",
            summary=summary,
            importance=importance,
            data={
                "tool_scores": tool_scores,
                "regime": regime,
                "outcome": outcome,
                "pnl": pnl,
            }
        )

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        if not self.enabled:
            return ToolResult(tool_name=self.name, score=0.0, confidence=0.0)

        now = time.time()
        if self.last_result and (now - self.last_run) < 30:
            return self.last_result

        tool_scores = ctx.get("prev_scores", {})
        regime = ctx.get("regime", "unknown")

        # Build a recall query from the current setup
        active_tools = [k for k, v in tool_scores.items() if abs(v) > 0.2]
        query = f"{regime} {' '.join(active_tools)}"

        recalled = self.memory.recall(query, top=5)

        if not recalled:
            # No similar past setups — neutral
            result = ToolResult(
                tool_name=self.name,
                score=0.0,
                confidence=0.2,
                features={
                    "recalled_count": 0,
                    "memory_size": len(self.memory),
                },
                metadata={"reasoning": "No similar past setups found."}
            )
        else:
            # Score based on how similar past trades performed
            wins = 0
            losses = 0
            total_pnl = 0.0
            for ep in recalled:
                pnl = ep.data.get("pnl", 0.0)
                outcome = ep.data.get("outcome", "")
                if "win" in outcome.lower() or pnl > 0:
                    wins += 1
                else:
                    losses += 1
                total_pnl += pnl

            total = wins + losses
            if total > 0:
                win_rate = wins / total
                # Score: positive if past similar trades were profitable
                score = (win_rate - 0.5) * 2  # scale to [-1, 1]
                confidence = min(1.0, 0.3 + 0.1 * total)
            else:
                score = 0.0
                confidence = 0.2

            result = ToolResult(
                tool_name=self.name,
                score=float(score),
                confidence=float(confidence),
                features={
                    "recalled_count": len(recalled),
                    "win_rate": wins / total if total else 0.0,
                    "total_pnl": total_pnl,
                    "memory_size": len(self.memory),
                },
                metadata={
                    "reasoning": f"Recalled {len(recalled)} similar setups. "
                                 f"Win rate: {wins}/{total}.",
                    "recalled_episodes": [
                        {"summary": ep.summary, "pnl": ep.data.get("pnl", 0)}
                        for ep in recalled
                    ],
                }
            )

        self.last_run = now
        self.last_result = result
        return result
