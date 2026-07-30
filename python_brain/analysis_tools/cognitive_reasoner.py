"""
cognitive_reasoner.py
=====================
NeuroSense-powered semantic reasoning tool.

Uses the neurosense knowledge graph and inference engine to build a
semantic model of market conditions and derive conclusions from them.
Unlike the LLM-based AIReasoner, this tool reasons locally with no API
calls — it learns facts from market structure and infers new knowledge
via forward-chaining rules.
"""
from __future__ import annotations
import time
from typing import Dict, Optional
import pandas as pd

from .base_tool import BaseTool, ToolResult


class CognitiveReasonerTool(BaseTool):
    """Semantic market reasoning using the neurosense knowledge graph.

    Learns facts like 'trending is_a regime', 'regime has momentum',
    'breakout can follow consolidation', then uses the inference engine
    to derive conclusions such as 'trending has momentum'.
    """
    name = "cognitive_reasoner"

    # Seed knowledge installed once on first instantiation
    _seeded = False

    def __init__(self):
        super().__init__()
        self.last_run = 0
        self.last_result: Optional[ToolResult] = None
        self._init_brain()

    def _init_brain(self):
        try:
            from neurosense import Brain
            self.brain = Brain(name="market-reasoner")
            self.enabled = True
        except Exception:
            self.brain = None
            self.enabled = False
            return

        if not CognitiveReasonerTool._seeded:
            self.brain.read(
                "A trend is a regime. A range is a regime. "
                "A breakout is a regime. A trend has momentum. "
                "A range has low volatility. A breakout has high volatility. "
                "Momentum can drive price. Volatility can expand range. "
                "A trend can break. A range can break. "
                "Consolidation can precede breakout. "
                "Liquidity can fuel breakout. "
                "A trend is_a regime. A range is_a regime. "
                "A breakout is_a regime."
            )
            CognitiveReasonerTool._seeded = True

    def _market_facts(self, tool_scores: Dict[str, float]) -> str:
        """Translate technical scores into English facts for the brain."""
        facts = []
        if tool_scores.get("market_structure", 0) > 0.3:
            facts.append("market is_a trend.")
        elif tool_scores.get("market_structure", 0) < -0.3:
            facts.append("market is_a downtrend.")
        else:
            facts.append("market is_a range.")

        if tool_scores.get("volatility_bands", 0) > 0.5:
            facts.append("market has high volatility.")
        else:
            facts.append("market has low volatility.")

        if tool_scores.get("momentum_volume", 0) > 0.3:
            facts.append("market has momentum.")

        if tool_scores.get("liquidity", 0) > 0.3:
            facts.append("market has liquidity.")

        return " ".join(facts)

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        if not self.enabled:
            return ToolResult(tool_name=self.name, score=0.0, confidence=0.0)

        now = time.time()
        if self.last_result and (now - self.last_run) < 30:
            return self.last_result

        tool_scores = ctx.get("prev_scores", {})
        price = ctx.get("price", 0.0)
        symbol = ctx.get("symbol", "Unknown")

        # Feed current market state as facts
        market_text = self._market_facts(tool_scores)
        self.brain.read(market_text)

        # Reason about what the market can do
        can_do = self.brain.reason("market", "can")
        has = self.brain.reason("market", "has")

        # Derive a sentiment score from the reasoning
        bullish_signals = 0
        bearish_signals = 0

        if "momentum" in can_do or "momentum" in has:
            ms = tool_scores.get("market_structure", 0)
            if ms > 0:
                bullish_signals += 1
            elif ms < 0:
                bearish_signals += 1

        if "high volatility" in has:
            # High volatility is neutral — could go either way
            pass

        if "liquidity" in has:
            bullish_signals += 0.5

        total = bullish_signals + bearish_signals
        if total > 0:
            score = (bullish_signals - bearish_signals) / total
        else:
            score = 0.0

        confidence = min(1.0, 0.4 + 0.1 * (len(can_do) + len(has)))

        reasoning = f"market can: {', '.join(can_do) or 'nothing'}; market has: {', '.join(has) or 'nothing'}"

        result = ToolResult(
            tool_name=self.name,
            score=float(score),
            confidence=float(confidence),
            features={
                "cognitive_can": can_do,
                "cognitive_has": has,
                "fact_count": len(self.brain.knowledge),
            },
            metadata={"reasoning": reasoning}
        )
        self.last_run = now
        self.last_result = result
        return result
