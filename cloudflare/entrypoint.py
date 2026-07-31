#!/usr/bin/env python3
"""Cloudflare Container entrypoint for Veilcrean.

This process does two jobs:
1. Starts a lightweight HTTP server on PORT for Cloudflare Container routing.
2. Optionally supervises the Veilcrean Python Brain child process.

The HTTP surface is intentionally small because this project is primarily a
trading brain, not a public web application. Cloudflare Worker auth should guard
/status and /logs; /healthz can remain public for uptime checks.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
START_TIME = time.time()

_CHILD: subprocess.Popen[str] | None = None
_CHILD_STARTED_AT: float | None = None
_CHILD_RESTARTS = 0
_SHUTDOWN = threading.Event()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _safe_env_snapshot() -> dict[str, Any]:
    """Return non-secret runtime configuration for status output."""
    safe_keys = [
        "VEIL_ENV",
        "VEIL_CF_MODE",
        "VEIL_START_BRAIN",
        "VEIL_EXIT_ON_BRAIN_STOP",
        "VEIL_BRAIN_AUTO_RESTART",
        "VEIL_BRAIN_CMD",
        "VEIL_ZMQ_PUB",
        "VEIL_ZMQ_PULL",
        "VEIL_ZMQ_STATUS",
        "DERIV_APP_ID",
        "DERIV_ENABLED",
        "DERIV_IS_DEMO",
        "VEIL_LLM_PROVIDER",
        "VEIL_LLM_ENABLED",
        "PORT",
        "CLOUDFLARE_DEPLOYMENT_ID",
    ]
    secret_keys = [
        "DERIV_API_TOKEN",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "VEIL_TG_TOKEN",
        "VEIL_TG_CHAT",
        "VEIL_DISCORD_HOOK",
    ]
    return {
        "config": {key: os.getenv(key, "") for key in safe_keys},
        "secrets_present": {key: bool(os.getenv(key)) for key in secret_keys},
    }


def _child_status() -> dict[str, Any]:
    child = _CHILD
    if child is None:
        return {
            "configured_to_start": _bool_env("VEIL_START_BRAIN", True),
            "running": False,
            "pid": None,
            "returncode": None,
            "started_at": None,
            "uptime_seconds": None,
            "restarts": _CHILD_RESTARTS,
        }

    returncode = child.poll()
    running = returncode is None
    return {
        "configured_to_start": _bool_env("VEIL_START_BRAIN", True),
        "running": running,
        "pid": child.pid,
        "returncode": returncode,
        "started_at": _CHILD_STARTED_AT,
        "uptime_seconds": round(time.time() - _CHILD_STARTED_AT, 3) if _CHILD_STARTED_AT else None,
        "restarts": _CHILD_RESTARTS,
    }


def _tail(path: Path, lines: int = 200) -> list[str]:
    if not path.exists():
        return []
    lines = max(1, min(lines, 1000))
    # Efficient enough for rotated application logs; keeps endpoint dependency-free.
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return fh.readlines()[-lines:]


class Handler(BaseHTTPRequestHandler):
    server_version = "VeilcreanCloudflare/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, status: int, body: str, content_type: str = "text/plain; charset=utf-8") -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        status = _child_status()

        if path in {"/", "/status"}:
            http_status = 200
            if status["configured_to_start"] and not status["running"]:
                http_status = 503
            self._send_json(
                http_status,
                {
                    "app": "Veilcrean",
                    "mode": "cloudflare-container",
                    "server": {
                        "uptime_seconds": round(time.time() - START_TIME, 3),
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    "brain": status,
                    "runtime": _safe_env_snapshot(),
                    "notes": [
                        "The Cloudflare Container HTTP endpoint is for health/status only.",
                        "MT5/ZMQ trading still needs a reachable broker/EA bridge; localhost endpoints only work inside the same container.",
                        "Container filesystem is ephemeral. Persist journal/models/logs externally for live production.",
                    ],
                },
            )
            return

        if path in {"/health", "/healthz", "/livez"}:
            self._send_json(200, {"ok": True, "uptime_seconds": round(time.time() - START_TIME, 3)})
            return

        if path == "/readyz":
            ready = (not status["configured_to_start"]) or bool(status["running"])
            self._send_json(200 if ready else 503, {"ready": ready, "brain": status})
            return

        if path == "/logs":
            query = parse_qs(parsed.query)
            lines = _int_env("VEIL_LOG_LINES", 200)
            if "lines" in query and query["lines"]:
                try:
                    lines = int(query["lines"][0])
                except ValueError:
                    pass
            content = "".join(_tail(ROOT / "logs" / "veilcrean.log", lines=lines))
            if not content:
                content = "No Veilcrean log file exists yet. Check Cloudflare container logs for stdout/stderr.\n"
            self._send_text(200, content)
            return

        if path == "/metrics":
            running = 1 if status["running"] else 0
            uptime = round(time.time() - START_TIME, 3)
            body = (
                "# HELP veilcrean_container_uptime_seconds Container supervisor uptime.\n"
                "# TYPE veilcrean_container_uptime_seconds gauge\n"
                f"veilcrean_container_uptime_seconds {uptime}\n"
                "# HELP veilcrean_brain_running Whether the Veilcrean brain child process is running.\n"
                "# TYPE veilcrean_brain_running gauge\n"
                f"veilcrean_brain_running {running}\n"
                "# HELP veilcrean_brain_restarts Child process restart count.\n"
                "# TYPE veilcrean_brain_restarts counter\n"
                f"veilcrean_brain_restarts {_CHILD_RESTARTS}\n"
            )
            self._send_text(200, body)
            return

        self._send_json(404, {"error": "not_found", "path": path})


def _start_brain() -> subprocess.Popen[str]:
    global _CHILD_STARTED_AT
    cmd = os.getenv("VEIL_BRAIN_CMD", "python -m python_brain.main")
    args = shlex.split(cmd)
    if not args:
        raise RuntimeError("VEIL_BRAIN_CMD produced an empty command")
    print(f"Starting Veilcrean brain: {args}", flush=True)
    _CHILD_STARTED_AT = time.time()
    return subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        start_new_session=True,
    )


def _stop_child(timeout: float = 30.0) -> None:
    child = _CHILD
    if child is None or child.poll() is not None:
        return
    print("Stopping Veilcrean brain...", flush=True)
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        child.terminate()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if child.poll() is not None:
            return
        time.sleep(0.2)
    print("Veilcrean brain did not stop in time; killing.", flush=True)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except Exception:
        child.kill()


def _handle_signal(signum: int, _frame: Any) -> None:
    print(f"Received signal {signum}; shutting down supervisor.", flush=True)
    _SHUTDOWN.set()
    _stop_child()


def main() -> int:
    global _CHILD, _CHILD_RESTARTS

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    port = _int_env("PORT", 8080)
    start_brain = _bool_env("VEIL_START_BRAIN", True)
    auto_restart = _bool_env("VEIL_BRAIN_AUTO_RESTART", False)
    exit_on_stop = _bool_env("VEIL_EXIT_ON_BRAIN_STOP", True)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, name="health-server", daemon=True)
    thread.start()
    print(f"Veilcrean Cloudflare health server listening on 0.0.0.0:{port}", flush=True)

    if start_brain:
        _CHILD = _start_brain()
    else:
        print("VEIL_START_BRAIN=false; serving health/status without starting trading brain.", flush=True)

    exit_code = 0
    try:
        while not _SHUTDOWN.is_set():
            if _CHILD is not None:
                returncode = _CHILD.poll()
                if returncode is not None:
                    print(f"Veilcrean brain exited with code {returncode}.", flush=True)
                    exit_code = returncode
                    if auto_restart and not _SHUTDOWN.is_set():
                        _CHILD_RESTARTS += 1
                        time.sleep(2)
                        _CHILD = _start_brain()
                        continue
                    if exit_on_stop:
                        _SHUTDOWN.set()
                        break
                    _CHILD = None
            time.sleep(1)
    finally:
        _SHUTDOWN.set()
        _stop_child()
        server.shutdown()
        server.server_close()
        print("Veilcrean Cloudflare supervisor stopped.", flush=True)

    return int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
