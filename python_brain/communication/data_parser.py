"""
data_parser.py
==============
Turns the raw JSON dict coming from the MT5 EA into a strongly-typed
``MarketSnapshot`` plus per-timeframe pandas DataFrames.

Output shape
------------
    MarketSnapshot(
        symbol     = "EURUSD",
        trigger    = "TICK" | "CANDLE" | "TRADE_EVENT",
        timestamp  = datetime,
        tick       = TickData(bid, ask, spread, volume),
        candles    = {"M1": DataFrame, "M5": DataFrame, ...},
        account    = AccountData(balance, equity, ...),
        positions  = [PositionData(...), ...],
    )
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- DTOs
@dataclass
class TickData:
    bid:    float
    ask:    float
    spread: float           # in points
    volume: int


@dataclass
class AccountData:
    balance:     float
    equity:      float
    margin_free: float
    margin_used: float
    profit:      float
    leverage:    int


@dataclass
class PositionData:
    ticket: int
    type:   str              # "BUY" | "SELL"
    symbol: str
    lots:   float
    open:   float
    sl:     float
    tp:     float
    profit: float


@dataclass
class MarketSnapshot:
    symbol:     str
    trigger:    str
    timestamp:  datetime
    tick:       Optional[TickData]               = None
    candles:    Dict[str, pd.DataFrame]          = field(default_factory=dict)
    account:    Optional[AccountData]            = None
    positions:  List[PositionData]               = field(default_factory=list)
    packet_type: str = "MARKET_DATA"


# --------------------------------------------------------------------------- parser
class DataParser:
    """Stateless JSON → MarketSnapshot parser."""

    # -------------------------------------------------------------- public
    def parse(self, raw: Dict[str, Any]) -> Optional[MarketSnapshot]:
        """Convert a JSON dict to a MarketSnapshot. Returns None on error."""
        if not raw or "type" not in raw:
            return None
        pkt_type = raw["type"]
        if pkt_type == "HEARTBEAT":
            # we keep these in zmq_server, parser doesn't need to do anything
            return None
        if pkt_type == "ACCOUNT_UPDATE":
            return self._parse_account_update(raw)
        if pkt_type == "EXEC_RESULT":
            # trade-execution result — handled by the trade loop, not the parser
            return None
        if pkt_type != "MARKET_DATA":
            return None
        return self._parse_market_data(raw)

    # -------------------------------------------------------------- internal
    def _parse_market_data(self, raw: Dict[str, Any]) -> MarketSnapshot:
        snap = MarketSnapshot(
            symbol    = raw.get("symbol", ""),
            trigger   = raw.get("trigger", "TICK"),
            timestamp = self._ts(raw.get("timestamp")),
            tick      = self._parse_tick(raw.get("tick", {})),
            candles   = self._parse_candles(raw.get("candles", {})),
            account   = self._parse_account(raw.get("account", {})),
            positions = self._parse_positions(raw.get("positions", [])),
            packet_type="MARKET_DATA",
        )
        return snap

    def _parse_account_update(self, raw: Dict[str, Any]) -> MarketSnapshot:
        return MarketSnapshot(
            symbol    = raw.get("symbol", ""),
            trigger   = raw.get("trigger", "ACCOUNT_UPDATE"),
            timestamp = self._ts(raw.get("timestamp")),
            candles   = {},     # account updates don't include candles
            account   = self._parse_account(raw.get("account", {})),
            positions = self._parse_positions(raw.get("positions", [])),
            packet_type="ACCOUNT_UPDATE",
        )

    @staticmethod
    def _ts(v) -> datetime:
        if v is None: return datetime.utcnow()
        try:    return datetime.utcfromtimestamp(int(v))
        except Exception: return datetime.utcnow()

    def _parse_tick(self, t: Dict[str, Any]) -> TickData:
        return TickData(
            bid    = float(t.get("bid", 0.0)),
            ask    = float(t.get("ask", 0.0)),
            spread = float(t.get("spread", 0.0)),
            volume = int(t.get("volume", 0)),
        )

    def _parse_account(self, a: Dict[str, Any]) -> AccountData:
        return AccountData(
            balance     = float(a.get("balance", 0.0)),
            equity      = float(a.get("equity", 0.0)),
            margin_free = float(a.get("margin_free", 0.0)),
            margin_used = float(a.get("margin_used", 0.0)),
            profit      = float(a.get("profit", 0.0)),
            leverage    = int(a.get("leverage", 100)),
        )

    def _parse_positions(self, arr: List[Dict[str, Any]]) -> List[PositionData]:
        out = []
        for p in arr or []:
            out.append(PositionData(
                ticket = int(p.get("ticket", 0)),
                type   = str(p.get("type", "")),
                symbol = str(p.get("symbol", "")),
                lots   = float(p.get("lots", 0.0)),
                open   = float(p.get("open", 0.0)),
                sl     = float(p.get("sl", 0.0)),
                tp     = float(p.get("tp", 0.0)),
                profit = float(p.get("profit", 0.0)),
            ))
        return out

    def _parse_candles(self, c: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for tf, arr in c.items():
            if not arr: continue
            try:
                df = pd.DataFrame(arr)
                # Normalize column names
                rename = {"O":"open","H":"high","L":"low","C":"close","V":"volume","t":"time"}
                df = df.rename(columns=rename)
                # Some EA versions use tick_volume — keep both
                if "volume" not in df.columns and "tick_volume" in df.columns:
                    df["volume"] = df["tick_volume"]
                # Build datetime index
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                    df = df.set_index("time")
                df = df.sort_index()
                # Drop incomplete / zero-width candles
                df = df[(df["high"] >= df["low"]) & (df["open"] > 0)]
                out[tf] = df
            except Exception:
                continue
        return out


# --------------------------------------------------------------------------- helpers
def mid_price(snap: MarketSnapshot) -> float:
    """Mid price from current tick (or last close if no tick)."""
    if snap.tick is not None and snap.tick.bid > 0:
        return (snap.tick.bid + snap.tick.ask) / 2.0
    for tf in ("M1", "M5", "M15", "H1", "D1"):
        df = snap.candles.get(tf)
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
    return 0.0
