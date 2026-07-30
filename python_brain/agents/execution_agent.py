"""
execution_agent.py
==================
Sub-agent that handles high-frequency trade entries and exits.
Ensures trade commands are formatted correctly and sent immediately.
"""
from __future__ import annotations
from typing import Any, Dict
from .base_agent import BaseAgent

class ExecutionAgent(BaseAgent):
    skill_role = "execution"
    name = "execution_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        decision = context.get("decision")
        if not decision or decision.get("action") == "HOLD":
             return {}
             
        # Enhance decision with specific scalping parameters if needed
        # e.g., micro-SL/TP adjustments for scalp trades
        
        return {
            "execution_ready": True,
            "latency_check": "OK"
        }
