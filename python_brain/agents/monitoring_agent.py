"""
monitoring_agent.py
===================
Agent that publishes real-time brain metrics to a ZeroMQ status socket.
Feeds the dashboard.
"""
from __future__ import annotations
from typing import Any, Dict
import time
from .base_agent import BaseAgent

class MonitoringAgent(BaseAgent):
    name = "monitoring_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Gathers info from all other agents and prepares a "Dashboard Bundle"
        snapshot = context.get("snapshot")
        decision = context.get("decision", {})
        sentiment = context.get("sentiment", {})
        
        bundle = {
            "ts": time.time(),
            "symbol": snapshot.symbol if snapshot else "N/A",
            "action": decision.get("action", "HOLD"),
            "confidence": decision.get("confidence", 0.0),
            "regime": decision.get("regime", "UNKNOWN"),
            "sentiment": sentiment.get("sentiment_summary", "Neutral"),
            "risk_ok": context.get("risk_ok", False)
        }
        
        # In a real system, we'd publish this to ZMQ here
        return {"dashboard_bundle": bundle}
