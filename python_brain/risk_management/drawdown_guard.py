"""
drawdown_guard.py
=================
Tracks peak equity and daily P&L. Raises the kill switch when hard
limits are breached.
"""
from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from ..config import RISK_CFG


class DrawdownGuard:
    def __init__(self):
        self.peak_equity: Optional[float] = None
        self.start_of_day_equity: Optional[float] = None
        self.last_day_key: Optional[str] = None
        self.kill_switch: bool = False
        self.kill_reason: str = ""

    # ------------------------------------------------------------------ API
    def update(self, equity: float) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        if self.last_day_key != today:
            self.start_of_day_equity = equity
            self.last_day_key = today

        # 1. Total drawdown check
        if self.peak_equity and self.peak_equity > 0:
            dd_pct = (self.peak_equity - equity) / self.peak_equity * 100.0
            if dd_pct >= RISK_CFG.max_total_drawdown_pct:
                self.kill_switch = True
                self.kill_reason = f"total drawdown {dd_pct:.2f}% > {RISK_CFG.max_total_drawdown_pct}%"
                return

        # 2. Daily loss check
        if self.start_of_day_equity and self.start_of_day_equity > 0:
            daily_pct = (self.start_of_day_equity - equity) / self.start_of_day_equity * 100.0
            if daily_pct >= RISK_CFG.max_daily_loss_pct:
                self.kill_switch = True
                self.kill_reason = f"daily loss {daily_pct:.2f}% > {RISK_CFG.max_daily_loss_pct}%"
                return

    def reset_kill(self) -> None:
        self.kill_switch = False
        self.kill_reason = ""

    def status(self) -> dict:
        return {
            "kill_switch": self.kill_switch,
            "kill_reason": self.kill_reason,
            "peak_equity": self.peak_equity,
            "start_of_day_equity": self.start_of_day_equity,
        }
