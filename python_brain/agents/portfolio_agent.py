"""
portfolio_agent.py
==================
Agent that looks at the big picture: account health, multiple pairs,
and long-term equity curve.
"""
from __future__ import annotations
from typing import Any, Dict
from .base_agent import BaseAgent

class PortfolioAgent(BaseAgent):
    name = "portfolio_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = context.get("snapshot")
        if not snapshot or not snapshot.account:
            return {}
            
        balance = snapshot.account.balance
        equity  = snapshot.account.equity
        drawdown = (balance - equity) / balance if balance > 0 else 0
        
        return {
            "account_health": "GOOD" if drawdown < 0.05 else "WARNING" if drawdown < 0.1 else "CRITICAL",
            "current_drawdown_pct": drawdown * 100
        }
