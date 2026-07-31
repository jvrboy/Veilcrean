"""Optional Veilcrean -> Vercel -> Supabase bridge.

The full Veilcrean brain is long-running and should run near MT5/broker access.
Vercel hosts the lightweight API layer. When VEIL_VERCEL_API_URL and
VEIL_API_KEY are configured, this bridge posts status/trade events to that API,
which then stores them in Supabase.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from ..utils.logger import get_logger

log = get_logger("vercel_supabase_bridge")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


class VercelSupabaseBridge:
    """Small, failure-safe HTTP client for the Vercel API."""

    def __init__(self) -> None:
        self.base_url = (os.getenv("VEIL_VERCEL_API_URL") or os.getenv("VERCEL_API_URL") or "").rstrip("/")
        self.api_key = os.getenv("VEIL_API_KEY") or ""
        self.timeout = float(os.getenv("VEIL_VERCEL_TIMEOUT_SECONDS", "4"))
        self.enabled = bool(self.base_url and self.api_key)
        self._last_error_ts = 0.0
        if self.enabled:
            log.info("Vercel/Supabase bridge enabled")

    def _warn_throttled(self, message: str) -> None:
        now = time.time()
        if now - self._last_error_ts > 60:
            log.warning(message)
            self._last_error_ts = now

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
        query: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        if not self.enabled:
            return None

        qs = urllib.parse.urlencode(query or {}, doseq=True)
        url = f"{self.base_url}{path}" + (f"?{qs}" if qs else "")
        body = None if payload is None else json.dumps(_jsonable(payload)).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "content-type": "application/json",
                "accept": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")[:500]
            self._warn_throttled(f"Vercel/Supabase bridge HTTP {exc.code}: {raw}")
        except Exception as exc:
            self._warn_throttled(f"Vercel/Supabase bridge failed: {exc}")
        return None

    def _get(self, path: str, query: Optional[dict[str, Any]] = None) -> Optional[Any]:
        return self._request("GET", path, query=query)

    def _post(self, path: str, payload: dict[str, Any]) -> Optional[Any]:
        return self._request("POST", path, payload=payload)

    def send_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        symbol: Optional[str] = None,
        severity: str = "info",
    ) -> Optional[Any]:
        return self._post(
            "/api/events",
            {
                "event_type": event_type,
                "source": "veilcrean-brain",
                "symbol": symbol,
                "severity": severity,
                "payload": payload,
            },
        )

    def send_status(self, status: dict[str, Any]) -> Optional[Any]:
        return self._post(
            "/api/status",
            {
                "source": "veilcrean-brain",
                "status": status,
                "win_rate": status.get("win_rate"),
                "profit_factor": status.get("profit_factor"),
                "sharpe": status.get("sharpe"),
                "max_dd_pct": status.get("max_dd_pct"),
                "trades": status.get("trades"),
                "threshold": status.get("threshold"),
                "kill_switch": status.get("kill_switch"),
                "regime": status.get("regime"),
                "confidence": status.get("confidence"),
            },
        )

    def send_signal(self, signal: dict[str, Any]) -> Optional[Any]:
        return self._post(
            "/api/signals",
            {
                "source": "veilcrean-brain",
                "signal_id": signal.get("signal_id") or signal.get("id"),
                "symbol": signal.get("symbol", "UNKNOWN"),
                "timeframe": signal.get("timeframe"),
                "action": signal.get("action") or signal.get("direction") or "UNKNOWN",
                "confidence": signal.get("confidence"),
                "price": signal.get("price"),
                "payload": signal,
            },
        )

    def send_trade(self, trade: dict[str, Any]) -> Optional[Any]:
        return self._post(
            "/api/trades",
            {
                "source": "veilcrean-brain",
                "trade_id": trade.get("trade_id") or trade.get("id"),
                "symbol": trade.get("symbol", "UNKNOWN"),
                "direction": trade.get("direction") or trade.get("action"),
                "status": trade.get("status", "open"),
                "opened_at": trade.get("opened_at"),
                "closed_at": trade.get("closed_at"),
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "sl": trade.get("sl"),
                "tp": trade.get("tp"),
                "lots": trade.get("lots") or trade.get("lot_size"),
                "pnl": trade.get("pnl"),
                "pnl_pct": trade.get("pnl_pct"),
                "confidence": trade.get("confidence"),
                "payload": trade,
            },
        )

    def get_pending_commands(self, limit: int = 10) -> list[dict[str, Any]]:
        data = self._get("/api/commands/pending", {"limit": max(1, min(int(limit), 50))})
        return data if isinstance(data, list) else []

    def complete_command(self, command_id: str, status: str, result: dict[str, Any]) -> Optional[Any]:
        return self._post(
            f"/api/commands/{command_id}/complete",
            {
                "status": status,
                "result": result,
            },
        )
