"""
broker_agent.py
===============
Agent 13 — Broker Abstraction Agent

Determines whether to route trade commands to MT5 (via ZMQ) or
directly to the Deriv API.
"""
from __future__ import annotations
from typing import Any, Dict
from .base_agent import BaseAgent
from ..config import DERIV_CFG

class BrokerAgent(BaseAgent):
    skill_role = "broker"
    name = "broker_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Logic to choose the best broker/platform for the current trade
        # For now, we prefer Deriv if enabled and available, otherwise MT5.
        
        preferred_platform = "MT5"
        if DERIV_CFG.enabled and DERIV_CFG.api_token:
            preferred_platform = "DERIV"
            
        return {
            "platform": preferred_platform,
            "broker_status": "ONLINE"
        }
