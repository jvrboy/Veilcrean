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

    def update_rows(self, table: str, *, query: dict[str, Any], patch: dict[str, Any]) -> Any:
        params: dict[str, Any] = {"select": "*"}
        params.update(query)
        return self._request("PATCH", table, query=params, body=patch)


supabase = SupabaseREST()


class TelegramBot:
    """Minimal Telegram Bot API client using only stdlib."""

    def __init__(self) -> None:
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.timeout = float(os.getenv("TELEGRAM_TIMEOUT_SECONDS", "8"))

    @property
    def configured(self) -> bool:
        return bool(self.token)

    def api(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def send_message(self, chat_id: int | str, text: str, *, disable_web_page_preview: bool = True) -> Any:
        if not self.configured:
            return None
        # Telegram's limit is 4096 chars. Keep margin for safety.
        safe_text = text if len(text) <= 3900 else text[:3890] + "\n..."
        payload = {
            "chat_id": chat_id,
            "text": safe_text,
            "disable_web_page_preview": disable_web_page_preview,
        }
        request = urllib.request.Request(
            self.api("sendMessage"),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except Exception:
            # Never fail the webhook just because Telegram reply failed.
            return None


telegram = TelegramBot()


def _allowed_chat_ids() -> set[str]:
    raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
    return {part.strip() for part in raw.replace(";", ",").split(",") if part.strip()}


def _telegram_allowed(chat_id: int | str) -> bool:
    allowed = _allowed_chat_ids()
    return not allowed or str(chat_id) in allowed


def _telegram_secret_ok(request: Request) -> bool:
    expected = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
    if not expected:
        return True
    return request.headers.get("x-telegram-bot-api-secret-token") == expected


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt_num(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def _row_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    return payload if isinstance(payload, dict) else {}


def _fmt_status(rows: Any) -> str:
    if not rows:
        return "No Veilcrean status has been recorded yet."
    row = rows[0] if isinstance(rows, list) else rows
    status = row.get("status") if isinstance(row.get("status"), dict) else {}
    return "\n".join([
        "Veilcrean latest status",
        f"Time: {row.get('created_at', '—')}",
        f"Regime: {row.get('regime') or status.get('regime') or '—'}",
        f"Confidence: {_fmt_num(row.get('confidence') or status.get('confidence'))}",
        f"Win rate: {_fmt_num(row.get('win_rate') or status.get('win_rate'))}",
        f"Profit factor: {_fmt_num(row.get('profit_factor') or status.get('profit_factor'))}",
        f"Sharpe: {_fmt_num(row.get('sharpe') or status.get('sharpe'))}",
        f"Max DD %: {_fmt_num(row.get('max_dd_pct') or status.get('max_dd_pct'))}",
        f"Trades: {row.get('trades') or status.get('trades') or 0}",
        f"Kill switch: {row.get('kill_switch') if row.get('kill_switch') is not None else status.get('kill_switch', False)}",
    ])


def _fmt_signals(rows: Any) -> str:
    if not rows:
        return "No signals recorded yet."
    lines = ["Latest signals"]
    for row in rows[:10]:
        lines.append(
            f"{row.get('created_at', '—')} | {row.get('symbol', '—')} | "
            f"{row.get('action', '—')} | conf {_fmt_num(row.get('confidence'))} | price {_fmt_num(row.get('price'), 5)}"
        )
    return "\n".join(lines)


def _fmt_trades(rows: Any) -> str:
    if not rows:
        return "No trades recorded yet."
    lines = ["Latest trades"]
    for row in rows[:10]:
        lines.append(
            f"{row.get('created_at', '—')} | {row.get('symbol', '—')} | "
            f"{row.get('direction') or '—'} | {row.get('status', '—')} | "
            f"entry {_fmt_num(row.get('entry_price'), 5)} | pnl {_fmt_num(row.get('pnl'))}"
        )
    return "\n".join(lines)


def _fmt_events(rows: Any) -> str:
    if not rows:
        return "No events recorded yet."
    lines = ["Latest events"]
    for row in rows[:10]:
        payload = _row_payload(row)
        symbol = row.get("symbol") or payload.get("symbol") or "—"
        lines.append(f"{row.get('created_at', '—')} | {row.get('event_type', 'event')} | {symbol} | {row.get('severity', 'info')}")
    return "\n".join(lines)


def _fmt_commands(rows: Any) -> str:
    if not rows:
        return "No commands queued yet."
    lines = ["Recent command queue"]
    for row in rows[:10]:
        lines.append(
            f"{row.get('created_at', '—')} | {row.get('command', '—')} {row.get('args') or ''} | {row.get('status', '—')}"
        )
    return "\n".join(lines)


def _help_text(chat_id: Optional[int | str] = None) -> str:
    chat_line = f"\nYour chat id: {chat_id}" if chat_id is not None else ""
    return (
        "Veilcrean Telegram commands\n"
        "/status - latest bot status\n"
        "/signals [n] - latest signals\n"
        "/trades [n] - latest trades\n"
        "/events [n] - latest events\n"
        "/queue - recent queued commands\n"
        "/pause [reason] - queue pause command\n"
        "/resume [reason] - queue resume command\n"
        "/flatten [symbol] - queue flatten-all command; execution requires brain opt-in\n"
        "/command NAME [args] - queue a custom command\n"
        "/id - show your Telegram chat id\n"
        "/help - this help"
        f"{chat_line}"
    )


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


class CommandIn(BaseModel):
    command: str = Field(..., max_length=80)
    args: Optional[str] = Field(default=None, max_length=500)
    source: str = Field(default="api", max_length=80)
    chat_id: Optional[str] = Field(default=None, max_length=80)
    username: Optional[str] = Field(default=None, max_length=120)
    payload: dict[str, Any] = Field(default_factory=dict)


class CommandCompleteIn(BaseModel):
    status: str = Field(default="processed", max_length=40)
    result: dict[str, Any] = Field(default_factory=dict)


class IngestIn(BaseModel):
    kind: str = Field(default="event", description="event, status, signal, trade, or command")
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
        "telegram_configured": telegram.configured,
        "telegram_allowed_chat_ids_configured": bool(_allowed_chat_ids()),
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
        "telegram_bot_token_present": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "telegram_webhook_secret_present": bool(os.getenv("TELEGRAM_WEBHOOK_SECRET")),
        "telegram_allowed_chat_ids_present": bool(os.getenv("TELEGRAM_ALLOWED_CHAT_IDS")),
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


@app.post("/api/commands", dependencies=[Depends(require_api_key)])
async def create_command(command: CommandIn) -> Any:
    row = command.model_dump(mode="json", exclude_none=True)
    row["command"] = row["command"].upper().strip()
    row.setdefault("status", "pending")
    return supabase.insert("bot_commands", row)


@app.get("/api/commands", dependencies=[Depends(maybe_require_read_key)])
async def list_commands(
    limit: int = Query(default=50, ge=1, le=500),
    status: Optional[str] = Query(default=None, max_length=40),
) -> Any:
    query = {"status": f"eq.{status}"} if status else None
    return supabase.list_rows("bot_commands", limit=limit, query=query)


@app.get("/api/commands/pending", dependencies=[Depends(require_api_key)])
async def pending_commands(limit: int = Query(default=10, ge=1, le=50)) -> Any:
    return supabase.list_rows("bot_commands", limit=limit, query={"status": "eq.pending"})


@app.post("/api/commands/{command_id}/complete", dependencies=[Depends(require_api_key)])
async def complete_command(command_id: str, result: CommandCompleteIn) -> Any:
    patch = result.model_dump(mode="json", exclude_none=True)
    patch["processed_at"] = _utc_now()
    return supabase.update_rows("bot_commands", query={"id": f"eq.{command_id}"}, patch=patch)


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
    if kind == "command":
        return await create_command(CommandIn(**data))
    return await create_event(EventIn(event_type=kind or "event", payload=data))


async def _queue_telegram_command(
    *,
    command: str,
    args: str,
    chat_id: int | str,
    username: Optional[str],
    raw_update: dict[str, Any],
) -> Any:
    return await create_command(
        CommandIn(
            command=command.upper().strip(),
            args=args.strip() or None,
            source="telegram",
            chat_id=str(chat_id),
            username=username,
            payload={"telegram_update": raw_update},
        )
    )


@app.get("/api/telegram/info", dependencies=[Depends(maybe_require_read_key)])
async def telegram_info() -> dict[str, Any]:
    return {
        "telegram_configured": telegram.configured,
        "webhook_secret_configured": bool(os.getenv("TELEGRAM_WEBHOOK_SECRET")),
        "allowed_chat_ids_configured": bool(_allowed_chat_ids()),
        "webhook_endpoint": "/api/telegram/webhook",
    }


@app.post("/api/telegram/webhook")
async def telegram_webhook(request: Request) -> dict[str, Any]:
    if not _telegram_secret_ok(request):
        raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret.")

    update = await request.json()
    message = update.get("message") or update.get("edited_message") or {}
    chat = message.get("chat") or {}
    from_user = message.get("from") or {}
    chat_id = chat.get("id")
    username = from_user.get("username") or from_user.get("first_name")
    text = (message.get("text") or "").strip()

    if chat_id is None:
        return {"ok": True, "ignored": "no_chat"}

    if not _telegram_allowed(chat_id):
        telegram.send_message(chat_id, f"This chat is not authorized. Chat id: {chat_id}")
        return {"ok": True, "authorized": False}

    if not text.startswith("/"):
        telegram.send_message(chat_id, _help_text(chat_id))
        return {"ok": True, "handled": "help"}

    parts = text.split(maxsplit=1)
    command = parts[0].split("@", 1)[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    try:
        if command in {"/start", "/help"}:
            reply = _help_text(chat_id)
        elif command == "/id":
            reply = f"Your Telegram chat id is: {chat_id}"
        elif command == "/health":
            h = await health()
            reply = "Veilcrean API health\n" + "\n".join(f"{k}: {v}" for k, v in h.items())
        elif command == "/status":
            reply = _fmt_status(await latest_status(limit=1))
        elif command == "/signals":
            limit = int(args.strip() or "5")
            reply = _fmt_signals(await latest_signals(limit=max(1, min(limit, 10))))
        elif command == "/trades":
            limit = int(args.strip() or "5")
            reply = _fmt_trades(await list_trades(limit=max(1, min(limit, 10))))
        elif command == "/events":
            limit = int(args.strip() or "5")
            reply = _fmt_events(await list_events(limit=max(1, min(limit, 10))))
        elif command == "/queue":
            reply = _fmt_commands(await list_commands(limit=10, status=None))
        elif command == "/pause":
            await _queue_telegram_command(command="PAUSE", args=args, chat_id=chat_id, username=username, raw_update=update)
            reply = "Queued PAUSE command. The brain must have command polling enabled to act on it."
        elif command == "/resume":
            await _queue_telegram_command(command="RESUME", args=args, chat_id=chat_id, username=username, raw_update=update)
            reply = "Queued RESUME command. The brain must have command polling enabled to act on it."
        elif command == "/flatten":
            await _queue_telegram_command(command="FLATTEN_ALL", args=args, chat_id=chat_id, username=username, raw_update=update)
            reply = "Queued FLATTEN_ALL command. Execution requires VEIL_ENABLE_REMOTE_TRADE_COMMANDS=true on the brain host."
        elif command == "/command":
            if not args.strip():
                reply = "Usage: /command NAME optional args"
            else:
                cmd_parts = args.split(maxsplit=1)
                custom_command = cmd_parts[0].upper()
                custom_args = cmd_parts[1] if len(cmd_parts) > 1 else ""
                await _queue_telegram_command(command=custom_command, args=custom_args, chat_id=chat_id, username=username, raw_update=update)
                reply = f"Queued {custom_command}."
        else:
            reply = "Unknown command.\n\n" + _help_text(chat_id)
    except HTTPException as exc:
        reply = f"API error: {exc.detail}"
    except Exception as exc:
        if _env_bool("DEBUG_ERRORS", False):
            reply = f"Command failed: {exc}\n{traceback.format_exc()}"
        else:
            reply = f"Command failed: {exc}"

    telegram.send_message(chat_id, reply)
    return {"ok": True, "command": command}


# Vercel expects an exported ASGI variable named `app`.
