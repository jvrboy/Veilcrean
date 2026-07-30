"""
risk_officer.py
===============
Hard safety and risk validation agent.
"""
from __future__ import annotations
from typing import Any, Dict

from .base_agent import BaseAgent
from ..risk_management.drawdown_guard import DrawdownGuard
from ..risk_management.exposure_manager import ExposureManager
from ..config import RISK_CFG

class RiskOfficerAgent(BaseAgent):
    skill_role = "risk_officer"
    name = "risk_officer"

    def __init__(self):
        super().__init__()
        self.guard = DrawdownGuard()
        self.exposure = ExposureManager()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = context.get("snapshot")
        decision = context.get("decision") # Proposed decision from Strategist
        
        if snapshot is None: return {"risk_ok": False, "reason": "No snapshot"}
        
        # 1. Heartbeat check
        # (This is handled in the coordinator for now, but could be here)
        
        # 2. Drawdown / Daily Loss
        if snapshot.account:
            self.guard.update(snapshot.account.equity)
            if self.guard.kill_switch:
                return {"risk_ok": False, "reason": f"KILL SWITCH: {self.guard.kill_reason}"}
        
        # 3. Spread
        if snapshot.tick and snapshot.tick.spread > RISK_CFG.max_spread_points:
            return {"risk_ok": False, "reason": f"Spread too wide: {snapshot.tick.spread}"}
            
        # 4. Exposure
        if decision and decision.get("action") != "HOLD":
            self.exposure.sync([{"symbol": p.symbol, "type": p.type} for p in snapshot.positions])
            if not self.exposure.can_open(snapshot.symbol, decision["action"]):
                return {"risk_ok": False, "reason": "Exposure limit reached"}

        return {"risk_ok": True}
