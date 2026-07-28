"""
market_regime_agent.py
======================
Sub-agent that specializes in identifying the current market regime.
"""
from __future__ import annotations
from typing import Any, Dict
from .base_agent import BaseAgent

class MarketRegimeAgent(BaseAgent):
    name = "regime_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        decision = context.get("decision", {})
        regime = decision.get("regime", "UNKNOWN")
        
        # Logic to adjust strategy based on regime
        adjustments = {}
        if regime == "TRENDING":
            adjustments["tp_multiplier"] = 1.5
            adjustments["sl_tighten"] = False
        elif regime == "RANGING":
            adjustments["tp_multiplier"] = 0.8
            adjustments["sl_tighten"] = True
        elif regime == "VOLATILE":
            adjustments["risk_multiplier"] = 0.5
            
        return {
            "regime": regime,
            "regime_adjustments": adjustments
        }
