"""
db_manager.py
=============
Tiny SQLite wrapper. Single-connection, thread-safe enough for the
single-threaded Veilcrean brain loop.

All schema migrations in ``python_brain/database/migrations/*.sql`` are
applied automatically on open, so a fresh database (e.g. on a new machine
or in Google Colab) always has every table the brain expects.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


class DBManager:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._apply_migrations()

    # ------------------------------------------------------------------ schema
    def _apply_migrations(self) -> None:
        """Execute migrations/*.sql in filename order (idempotent, IF NOT EXISTS)."""
        if not _MIGRATIONS_DIR.is_dir():
            return
        for sql_file in sorted(_MIGRATIONS_DIR.glob("*.sql")):
            script = sql_file.read_text(encoding="utf-8")
            if not script.strip():
                continue
            try:
                self._conn.executescript(script)
            except sqlite3.Error as e:
                raise RuntimeError(f"failed to apply migration {sql_file.name}: {e}") from e

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
