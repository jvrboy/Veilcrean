"""
strategist_agent.py
===================
Combines technical reports with machine learning models and AI reasoning
to form a high-level trading strategy.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List

from .base_agent import BaseAgent
from ..strategies.ict_po3 import ICTPO3Strategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.perpetual_scalper import PerpetualScalperStrategy

class StrategistAgent(BaseAgent):
    skill_role = "strategist"
    name = "strategist_agent"

    def __init__(self, decision_engine):
        super().__init__()
        self.engine = decision_engine
        self.strategies = [
            ICTPO3Strategy(),
            MeanReversionStrategy(),
            PerpetualScalperStrategy()
        ]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tech_report = context.get("technical_report")
        if tech_report is None:
            return {"action": "HOLD", "reason": "No technical report"}

        feature_vec = tech_report.get("feature_vector")
        if feature_vec is None:
            return {"action": "HOLD", "reason": "No feature vector"}

        # 1. Get Neural Network Decision
        decision = self.engine.decide(feature_vec)
        
        # 2. Check Rule-Based Strategies
        strategy_signals = []
        for strat in self.strategies:
            sig = strat.check_signal(context)
            if sig["action"] != "HOLD":
                strategy_signals.append(sig)
                
        # 3. Confluence Layer
        # If a rule-based strategy agrees with the NN, boost confidence significantly
        for sig in strategy_signals:
            if sig["action"] == decision["action"]:
                self.log.info(f"Confluence found: {sig['reason']} agrees with ML")
                decision["confidence"] = min(0.98, decision["confidence"] + 0.15)
                decision["strategy_match"] = sig["reason"]
            elif sig["action"] != "HOLD":
                # Conflict: if a strong strategy says opposite, lower confidence
                self.log.warning(f"Strategy conflict: {sig['reason']} says {sig['action']}")
                decision["confidence"] *= 0.7
        
        # 4. AI Reasoner Modulator
        ai_res = tech_report.get("tool_results", {}).get("AIReasonerTool")
        if ai_res and ai_res.score != 0:
             agreement = np.sign(ai_res.score) == np.sign(tech_report.get("aggregate_score", 0))
             if not agreement:
                 decision["confidence"] *= 0.8
        
        return decision
