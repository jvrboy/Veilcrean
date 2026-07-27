"""
db_manager.py
=============
Tiny SQLite wrapper. Single-connection, thread-safe enough for the
single-threaded Veilcrean brain loop.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional


class DBManager:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------ API
    def execute(self, sql: str, params: Optional[Iterable[Any]] = None):
        cur = self._conn.cursor()
        cur.execute(sql, params or [])
        return cur

    def executemany(self, sql: str, seq):
        cur = self._conn.cursor()
        cur.executemany(sql, seq)
        return cur

    def fetchall(self, sql: str, params: Optional[Iterable[Any]] = None):
        cur = self.execute(sql, params)
        return cur.fetchall()

    def fetchone(self, sql: str, params: Optional[Iterable[Any]] = None):
        cur = self.execute(sql, params)
        return cur.fetchone()

    def close(self) -> None:
        try: self._conn.close()
        except Exception: pass
