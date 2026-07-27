"""
alerts.py
=========
Telegram and Discord notifications. Optional — only fires if env vars
or config has credentials set.
"""
from __future__ import annotations
import json
from typing import Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None
    _REQUESTS_AVAILABLE = False

from ..config import ALERT_CFG
from .logger import get_logger

log = get_logger("alerts")


class Alerter:
    def __init__(self, cfg=None):
        self.cfg = cfg or ALERT_CFG

    # ------------------------------------------------------------------ public
    def send(self, message: str, level: str = "info") -> None:
        if self.cfg.enable_console:
            getattr(log, level if level in ("info","warning","error","debug") else "info")(message)
        if self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
            self._telegram(message)
        if self.cfg.discord_webhook_url:
            self._discord(message)

    def trade_open(self, symbol: str, direction: str, lots: float,
                   sl: float, tp: float, conf: float) -> None:
        if not self.cfg.notify_on_trade_open: return
        self.send(f"🟢 OPEN {direction} {symbol} lots={lots:.2f} "
                  f"sl={sl:.5f} tp={tp:.5f} conf={conf:.2f}", level="info")

    def trade_close(self, symbol: str, pnl: float, pnl_pct: float) -> None:
        if not self.cfg.notify_on_trade_close: return
        emoji = "✅" if pnl >= 0 else "❌"
        self.send(f"{emoji} CLOSE {symbol} pnl={pnl:+.2f} ({pnl_pct:+.2f}%)", level="info")

    def kill_switch(self, reason: str) -> None:
        if not self.cfg.notify_on_kill_switch: return
        self.send(f"🛑 KILL SWITCH — {reason}", level="error")

    def retrain(self, version: str, acc: float) -> None:
        if not self.cfg.notify_on_retrain: return
        self.send(f"🧠 Retrained {version} (acc={acc:.3f})", level="info")

    def error(self, msg: str) -> None:
        if not self.cfg.notify_on_errors: return
        self.send(f"⚠️ {msg}", level="error")

    # ------------------------------------------------------------------ private
    def _telegram(self, text: str) -> None:
        if not _REQUESTS_AVAILABLE: return
        try:
            url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
            requests.post(url, json={"chat_id": self.cfg.telegram_chat_id, "text": text}, timeout=5)
        except Exception as e:
            log.warning(f"telegram send failed: {e}")

    def _discord(self, text: str) -> None:
        if not _REQUESTS_AVAILABLE: return
        try:
            requests.post(self.cfg.discord_webhook_url,
                          json={"content": text}, timeout=5)
        except Exception as e:
            log.warning(f"discord send failed: {e}")
