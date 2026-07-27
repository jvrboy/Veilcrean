"""
zmq_server.py
=============
ZeroMQ wrapper that connects the Python brain to the MT5 EA.

Topology
--------
  EA ───PUB───► 5555  ◄──SUB── Python   (market data + heartbeats)
  EA ◄──PULL─── 5556  ───PUSH──► Python  (trade commands + results)

The server runs in the calling thread; the main loop is expected to
poll `receive_market_data()` on every iteration. We deliberately keep
the API simple and synchronous — multi-threading is unnecessary for
Veilcrean because all heavy work (NN inference) takes well under one
tick interval (typically 250 ms).
"""
from __future__ import annotations
import json
import time
from typing import Any, Dict, Optional, Callable

try:
    import zmq
    import zmq.asyncio  # noqa: F401  (kept for future async migration)
    _ZMQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    zmq = None
    _ZMQ_AVAILABLE = False

from ..config import ZMQ_CFG


class ZMQServer:
    """Synchronous ZMQ bridge to MT5 EA.

    Methods
    -------
    start()               — open sockets, bind / connect.
    stop()                — close sockets cleanly.
    receive_market_data() — non-blocking poll for an incoming packet.
    send_trade_command()  — send a dict as JSON to the EA.
    publish_status()      — broadcast internal state for dashboards.
    """

    def __init__(self,
                 pub_endpoint:  Optional[str] = None,
                 pull_endpoint: Optional[str] = None,
                 status_endpoint: Optional[str] = None):
        self.pub_endpoint    = pub_endpoint    or ZMQ_CFG.market_data_endpoint
        self.pull_endpoint   = pull_endpoint   or ZMQ_CFG.trade_command_endpoint
        self.status_endpoint = status_endpoint or ZMQ_CFG.brain_status_endpoint

        self._ctx: Optional[zmq.Context] = None
        self._sub: Optional[zmq.Socket] = None
        self._push: Optional[zmq.Socket] = None
        self._pub_status: Optional[zmq.Socket] = None

        # Last-seen heartbeat (used to detect EA death)
        self.last_heartbeat_ts: float = 0.0
        self.last_packet_ts: float = 0.0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Open all sockets. Safe to call once at startup."""
        if not _ZMQ_AVAILABLE:
            raise RuntimeError("pyzmq is not installed. Run `pip install pyzmq`.")
        self._ctx = zmq.Context.instance()
        # EA → Python  (we SUB to the EA's PUB)
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.RCVTIMEO, ZMQ_CFG.recv_timeout_ms)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")  # subscribe to all
        self._sub.connect(self.pub_endpoint)
        # Python → EA  (we PUSH, EA PULLs)
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.setsockopt(zmq.SNDTIMEO, ZMQ_CFG.send_timeout_ms)
        self._push.connect(self.pull_endpoint)
        # Brain status broadcast (PUB)
        self._pub_status = self._ctx.socket(zmq.PUB)
        self._pub_status.bind(self.status_endpoint)
        # give ZMQ a moment to wire up
        time.sleep(0.2)

    def stop(self) -> None:
        """Close sockets & terminate context."""
        for s in (self._sub, self._push, self._pub_status):
            try:
                if s is not None:
                    s.setsockopt(zmq.LINGER, 0)
                    s.close()
            except Exception:
                pass
        if self._ctx is not None:
            self._ctx.term()
        self._sub = self._push = self._pub_status = None
        self._ctx = None

    # ------------------------------------------------------------------ receive
    def receive_market_data(self) -> Optional[Dict[str, Any]]:
        """Return the next JSON packet from the EA, or None if nothing.

        Handles three packet types:
            MARKET_DATA  — full snapshot
            HEARTBEAT    — liveness ping
            ACCOUNT_UPDATE — smaller account/position update
        """
        if self._sub is None:
            return None
        try:
            raw = self._sub.recv_string(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        except zmq.ZMQError:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None

        self.last_packet_ts = time.time()
        if payload.get("type") == "HEARTBEAT":
            self.last_heartbeat_ts = time.time()
        return payload

    def heartbeat_is_alive(self, max_age_sec: float = 15.0) -> bool:
        """True if we have seen a heartbeat within `max_age_sec` seconds."""
        if self.last_heartbeat_ts == 0.0:
            # allow a grace period at boot
            return (time.time() - self.last_packet_ts) < max_age_sec
        return (time.time() - self.last_heartbeat_ts) < max_age_sec

    # ------------------------------------------------------------------ send
    def send_trade_command(self, command: Dict[str, Any]) -> bool:
        """Push a trade command to the EA. Returns True on success."""
        if self._push is None:
            return False
        try:
            self._push.send_string(json.dumps(command), flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    def publish_status(self, status: Dict[str, Any]) -> None:
        """Broadcast brain status to anyone subscribed to status_endpoint."""
        if self._pub_status is None:
            return
        try:
            self._pub_status.send_string(json.dumps(status), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
