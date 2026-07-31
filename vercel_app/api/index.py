"""Veilcrean Vercel + Supabase API.

This is intentionally lightweight and serverless-friendly. It does not run the
long-lived Veilcrean trading brain. Instead, it exposes authenticated HTTP
endpoints that store and retrieve Veilcrean status, signals, trades, and generic
bot events in Supabase via the PostgREST API.
"""
from __future__ import annotations

import json
import os
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


APP_VERSION = "1.0.0"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class SupabaseREST:
    """Tiny Supabase PostgREST client using only Python stdlib."""

    def __init__(self) -> None:
        self.url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
        self.schema = os.getenv("SUPABASE_SCHEMA", "public")
        self.timeout = float(os.getenv("SUPABASE_TIMEOUT_SECONDS", "12"))

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key)

    def _headers(self, prefer: str = "return=representation") -> dict[str, str]:
        return {
            "apikey": self.key,
            "authorization": f"Bearer {self.key}",
            "content-type": "application/json",
            "accept": "application/json",
            "accept-profile": self.schema,
            "content-profile": self.schema,
            "prefer": prefer,
        }

    def _request(
        self,
        method: str,
        table: str,
        *,
        query: Optional[dict[str, Any]] = None,
        body: Any = None,
        prefer: str = "return=representation",
    ) -> Any:
        if not self.configured:
            raise HTTPException(
                status_code=503,
                detail="Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Vercel.",
            )

        base = f"{self.url}/rest/v1/{urllib.parse.quote(table)}"
        qs = urllib.parse.urlencode(query or {}, doseq=True, safe="*,.():->")
        url = f"{base}?{qs}" if qs else base
        payload = None if body is None else json.dumps(_jsonable(body)).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method=method, headers=self._headers(prefer))

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw
            raise HTTPException(status_code=exc.code, detail={"supabase_error": parsed}) from exc
        except urllib.error.URLError as exc:
            raise HTTPException(status_code=502, detail=f"Supabase request failed: {exc}") from exc

    def insert(self, table: str, row: dict[str, Any]) -> Any:
        return self._request("POST", table, body=row)

    def list_rows(self, table: str, *, limit: int = 50, query: Optional[dict[str, Any]] = None) -> Any:
        safe_limit = max(1, min(int(limit), 500))
        params: dict[str, Any] = {"select": "*", "order": "created_at.desc", "limit": safe_limit}
        if query:
            params.update(query)
        return self._request("GET", table, query=params, prefer="")


supabase = SupabaseREST()

app = FastAPI(
    title="Veilcrean Vercel API",
    description="Serverless API layer for Veilcrean + Supabase.",
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["authorization", "content-type", "x-api-key"],
)


class EventIn(BaseModel):
    event_type: str = Field(default="event", max_length=80)
    source: str = Field(default="veilcrean", max_length=80)
    symbol: Optional[str] = Field(default=None, max_length=40)
    severity: str = Field(default="info", max_length=20)
    payload: dict[str, Any] = Field(default_factory=dict)


class StatusIn(BaseModel):
    source: str = Field(default="veilcrean", max_length=80)
    status: dict[str, Any] = Field(default_factory=dict)
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    sharpe: Optional[float] = None
    max_dd_pct: Optional[float] = None
    trades: Optional[int] = None
    threshold: Optional[float] = None
    kill_switch: Optional[bool] = None
    regime: Optional[str] = None
    confidence: Optional[float] = None


class SignalIn(BaseModel):
    signal_id: Optional[str] = Field(default=None, max_length=120)
    source: str = Field(default="veilcrean", max_length=80)
    symbol: str = Field(..., max_length=40)
    timeframe: Optional[str] = Field(default=None, max_length=20)
    action: str = Field(..., max_length=20)
    confidence: Optional[float] = None
    price: Optional[float] = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TradeIn(BaseModel):
    trade_id: Optional[str] = Field(default=None, max_length=120)
    source: str = Field(default="veilcrean", max_length=80)
    symbol: str = Field(..., max_length=40)
    direction: Optional[str] = Field(default=None, max_length=20)
    status: str = Field(default="open", max_length=20)
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    lots: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    confidence: Optional[float] = None
    payload: dict[str, Any] = Field(default_factory=dict)


class IngestIn(BaseModel):
    kind: str = Field(default="event", description="event, status, signal, or trade")
    data: dict[str, Any] = Field(default_factory=dict)


def require_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    expected = os.getenv("VEIL_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="VEIL_API_KEY is not configured in Vercel.")

    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()
    if bearer == expected or x_api_key == expected:
        return
    raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def maybe_require_read_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    if _env_bool("VEIL_PUBLIC_READ", False):
        return
    return require_api_key(authorization, x_api_key)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        raise exc
    if _env_bool("DEBUG_ERRORS", False):
        detail: Any = {"error": str(exc), "traceback": traceback.format_exc()}
    else:
        detail = "Internal server error"
    return JSONResponse(status_code=500, content={"detail": detail})


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <title>Veilcrean API</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <style>
          body { font-family: Inter, system-ui, sans-serif; margin: 40px; background: #080b12; color: #e7ecff; }
          .card { max-width: 860px; padding: 28px; border: 1px solid #26324a; border-radius: 18px; background: #101626; }
          code { color: #9be7ff; }
          a { color: #8ab4ff; }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>Veilcrean Vercel API</h1>
          <p>Serverless API layer connected to Supabase.</p>
          <ul>
            <li><code>GET /api/health</code></li>
            <li><code>POST /api/status</code></li>
            <li><code>POST /api/events</code></li>
            <li><code>POST /api/signals</code></li>
            <li><code>POST /api/trades</code></li>
            <li><a href="/api/docs">OpenAPI docs</a></li>
          </ul>
        </div>
      </body>
    </html>
    """


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "app": "Veilcrean Vercel API",
        "version": APP_VERSION,
        "time": _utc_now(),
        "supabase_configured": supabase.configured,
        "api_key_configured": bool(os.getenv("VEIL_API_KEY")),
        "public_read": _env_bool("VEIL_PUBLIC_READ", False),
    }


@app.get("/api/config", dependencies=[Depends(maybe_require_read_key)])
async def config() -> dict[str, Any]:
    return {
        "supabase_url_present": bool(os.getenv("SUPABASE_URL")),
        "supabase_key_present": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")),
        "schema": os.getenv("SUPABASE_SCHEMA", "public"),
        "cors_origins": cors_origins,
        "public_read": _env_bool("VEIL_PUBLIC_READ", False),
    }


@app.post("/api/events", dependencies=[Depends(require_api_key)])
async def create_event(event: EventIn) -> Any:
    row = event.model_dump(mode="json", exclude_none=True)
    return supabase.insert("bot_events", row)


@app.get("/api/events", dependencies=[Depends(maybe_require_read_key)])
async def list_events(limit: int = Query(default=50, ge=1, le=500)) -> Any:
    return supabase.list_rows("bot_events", limit=limit)


@app.post("/api/status", dependencies=[Depends(require_api_key)])
async def create_status(status: StatusIn) -> Any:
    data = status.model_dump(mode="json", exclude_none=True)
    merged = dict(data.get("status") or {})
    for key in ["win_rate", "profit_factor", "sharpe", "max_dd_pct", "trades", "threshold", "kill_switch", "regime", "confidence"]:
        if data.get(key) is not None:
            merged[key] = data[key]
    data["status"] = merged
    return supabase.insert("bot_status", data)


@app.get("/api/status/latest", dependencies=[Depends(maybe_require_read_key)])
async def latest_status(limit: int = Query(default=1, ge=1, le=100)) -> Any:
    return supabase.list_rows("bot_status", limit=limit)


@app.post("/api/signals", dependencies=[Depends(require_api_key)])
async def create_signal(signal: SignalIn) -> Any:
    return supabase.insert("trade_signals", signal.model_dump(mode="json", exclude_none=True))


@app.get("/api/signals/latest", dependencies=[Depends(maybe_require_read_key)])
async def latest_signals(limit: int = Query(default=20, ge=1, le=100)) -> Any:
    return supabase.list_rows("trade_signals", limit=limit)


@app.post("/api/trades", dependencies=[Depends(require_api_key)])
async def create_trade(trade: TradeIn) -> Any:
    return supabase.insert("trade_journal", trade.model_dump(mode="json", exclude_none=True))


@app.get("/api/trades", dependencies=[Depends(maybe_require_read_key)])
async def list_trades(limit: int = Query(default=50, ge=1, le=500)) -> Any:
    return supabase.list_rows("trade_journal", limit=limit)


@app.post("/api/ingest", dependencies=[Depends(require_api_key)])
async def ingest(item: IngestIn) -> Any:
    kind = item.kind.lower().strip()
    data = item.data or {}
    if kind == "status":
        return await create_status(StatusIn(**data))
    if kind == "signal":
        return await create_signal(SignalIn(**data))
    if kind == "trade":
        return await create_trade(TradeIn(**data))
    return await create_event(EventIn(event_type=kind or "event", payload=data))


# Vercel expects an exported ASGI variable named `app`.
