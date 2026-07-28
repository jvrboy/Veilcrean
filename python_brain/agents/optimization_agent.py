"""
optimization_agent.py
=====================
Agent 12 — Parameter Optimization Agent

Analyzes recent trade performance and suggests dynamic adjustments
to confidence thresholds and risk multipliers.
"""
from __future__ import annotations
from typing import Any, Dict
from .base_agent import BaseAgent
from ..config import SI_CFG

class OptimizationAgent(BaseAgent):
    name = "optimization_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Looks at the PerformanceTracker results from the context
        win_rate = context.get("win_rate", 0.5)
        
        # Dynamic threshold adjustment
        suggested_threshold = SI_CFG.confidence_threshold
        
        if win_rate < 0.45:
            # Low win rate: Be more selective
            suggested_threshold += 0.05
        elif win_rate > 0.65:
            # High win rate: Be more aggressive
            suggested_threshold -= 0.05
            
        suggested_threshold = max(0.6, min(0.95, suggested_threshold))
        
        return {
            "optimal_threshold": suggested_threshold,
            "risk_modifier": 1.2 if win_rate > 0.6 else 0.8 if win_rate < 0.4 else 1.0
        }
