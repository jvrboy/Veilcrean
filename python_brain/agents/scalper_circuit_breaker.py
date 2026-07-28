"""
scalper_circuit_breaker.py
==========================
Agent to prevent "Death Spirals" in perpetual scalpers.
If too many losses occur in a short window, it pauses the bot.
"""
from __future__ import annotations
from typing import Any, Dict
from .base_agent import BaseAgent
import time

class CircuitBreakerAgent(BaseAgent):
    name = "circuit_breaker"

    def __init__(self, max_losses: int = 5, window_sec: int = 3600):
        super().__init__()
        self.max_losses = max_losses
        self.window_sec = window_sec
        self.loss_timestamps = []
        self.paused_until = 0

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        
        # Check if we are currently in a mandatory pause
        if now < self.paused_until:
             return {"breaker_tripped": True, "pause_remaining": int(self.paused_until - now)}

        # Update losses from journal (this would be passed in context)
        # For now, we assume the coordinator passes recent results
        recent_trades = context.get("recent_trades", [])
        
        losses = [t for t in recent_trades if t.get('pnl', 0) < 0 and (now - t.get('closed_at', 0)) < self.window_sec]
        
        if len(losses) >= self.max_losses:
             self.paused_until = now + 1800 # 30 minute timeout
             self.log.error(f"CIRCUIT BREAKER TRIPPED: {len(losses)} losses in {self.window_sec}s")
             return {"breaker_tripped": True, "pause_remaining": 1800}

        return {"breaker_tripped": False}
