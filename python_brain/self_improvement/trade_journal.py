"""
trade_journal.py
================
SQLite-backed journal of every trade Veilcrean has ever taken.

This is the single source of truth for the self-improvement loop —
both the retrainer and the performance tracker read from it.
"""
from __future__ import annotations
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import JOURNAL_DB
from ..database.db_manager import DBManager


@dataclass
class TradeRecord:
    """One row in the trade_journal table."""
    trade_id:     str
    symbol:       str
    direction:    str
    opened_at:    float
    closed_at:    Optional[float] = None
    entry_price:  float = 0.0
    exit_price:   float = 0.0
    sl:           float = 0.0
    tp:           float = 0.0
    lots:         float = 0.0
    pnl:          float = 0.0
    pnl_pct:      float = 0.0
    r_achieved:   float = 0.0
    confidence:   float = 0.0
    regime:       str   = "UNKNOWN"
    session:      str   = ""
    weekday:      int   = 0
    strategy_tag: str   = ""
    feature_vec:  List[float] = field(default_factory=list)
    mae:          float = 0.0     # max adverse excursion
    mfe:          float = 0.0     # max favorable excursion
    is_win:       int   = 0       # 0/1
    notes:        str   = ""


class TradeJournal:
    """Thin wrapper over the SQLite trade_journal table."""

    def __init__(self, db_path: Path = JOURNAL_DB):
        self.db = DBManager(db_path)
        self._ensure_schema()

    # ------------------------------------------------------------------ schema
    def _ensure_schema(self) -> None:
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                trade_id      TEXT PRIMARY KEY,
                symbol        TEXT,
                direction     TEXT,
                opened_at     REAL,
                closed_at     REAL,
                entry_price   REAL,
                exit_price    REAL,
                sl            REAL,
                tp            REAL,
                lots          REAL,
                pnl           REAL,
                pnl_pct       REAL,
                r_achieved    REAL,
                confidence    REAL,
                regime        TEXT,
                session       TEXT,
                weekday       INTEGER,
                strategy_tag  TEXT,
                feature_vec   TEXT,
                mae           REAL,
                mfe           REAL,
                is_win        INTEGER,
                notes         TEXT
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_journal_opened ON trade_journal(opened_at)")

    # ------------------------------------------------------------------ API
    def open_trade(self, rec: TradeRecord) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO trade_journal
               (trade_id, symbol, direction, opened_at, entry_price, sl, tp, lots,
                confidence, regime, session, weekday, strategy_tag, feature_vec)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rec.trade_id, rec.symbol, rec.direction, rec.opened_at,
             rec.entry_price, rec.sl, rec.tp, rec.lots, rec.confidence,
             rec.regime, rec.session, rec.weekday, rec.strategy_tag,
             json.dumps(rec.feature_vec))
        )

    def close_trade(self, trade_id: str, exit_price: float, pnl: float, pnl_pct: float,
                    r_achieved: float, mae: float, mfe: float, is_win: int,
                    notes: str = "") -> None:
        self.db.execute(
            """UPDATE trade_journal
               SET closed_at=?, exit_price=?, pnl=?, pnl_pct=?, r_achieved=?,
                   mae=?, mfe=?, is_win=?, notes=?
             WHERE trade_id=?""",
            (time.time(), exit_price, pnl, pnl_pct, r_achieved, mae, mfe, is_win, notes, trade_id)
        )

    def all_closed(self) -> List[TradeRecord]:
        cur = self.db.execute(
            "SELECT * FROM trade_journal WHERE closed_at IS NOT NULL ORDER BY opened_at"
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        out: List[TradeRecord] = []
        for r in rows:
            d = dict(zip(cols, r))
            try:
                d["feature_vec"] = json.loads(d.get("feature_vec") or "[]")
            except Exception:
                d["feature_vec"] = []
            out.append(TradeRecord(**{k: d.get(k) for k in (
                "trade_id","symbol","direction","opened_at","closed_at","entry_price",
                "exit_price","sl","tp","lots","pnl","pnl_pct","r_achieved",
                "confidence","regime","session","weekday","strategy_tag","feature_vec",
                "mae","mfe","is_win","notes")}))
        return out

    def open_positions(self) -> List[TradeRecord]:
        cur = self.db.execute(
            "SELECT * FROM trade_journal WHERE closed_at IS NULL"
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        out: List[TradeRecord] = []
        for r in rows:
            d = dict(zip(cols, r))
            d["feature_vec"] = []
            out.append(TradeRecord(**{k: d.get(k) for k in (
                "trade_id","symbol","direction","opened_at","entry_price","exit_price",
                "sl","tp","lots","pnl","pnl_pct","r_achieved","confidence","regime",
                "session","weekday","strategy_tag","feature_vec","mae","mfe","is_win","notes")}))
        return out

    def count(self) -> int:
        cur = self.db.execute("SELECT COUNT(*) FROM trade_journal")
        return int(cur.fetchone()[0])

    def count_closed(self) -> int:
        cur = self.db.execute("SELECT COUNT(*) FROM trade_journal WHERE closed_at IS NOT NULL")
        return int(cur.fetchone()[0])

    def n_trades_since_last_train(self) -> int:
        cur = self.db.execute(
            "SELECT COUNT(*) FROM trade_journal WHERE closed_at > "
            "(SELECT COALESCE(MAX(last_train_ts), 0) FROM retrain_log)"
        )
        try:    return int(cur.fetchone()[0])
        except Exception: return 0

    def log_retrain(self, version: str, n_samples: int, acc: float) -> None:
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS retrain_log (
                last_train_ts REAL,
                version TEXT,
                n_samples INTEGER,
                acc REAL
            )"""
        )
        self.db.execute(
            "INSERT INTO retrain_log (last_train_ts, version, n_samples, acc) VALUES (?,?,?,?)",
            (time.time(), version, n_samples, acc)
        )
