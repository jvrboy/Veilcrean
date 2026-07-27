-- 001_initial.sql
-- Initial schema for the Veilcrean journal & logs.

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
);

CREATE INDEX IF NOT EXISTS idx_journal_opened ON trade_journal(opened_at);
CREATE INDEX IF NOT EXISTS idx_journal_closed ON trade_journal(closed_at);

CREATE TABLE IF NOT EXISTS retrain_log (
    last_train_ts REAL,
    version       TEXT,
    n_samples     INTEGER,
    acc           REAL
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    ts           REAL,
    symbol       TEXT,
    feature_vec  TEXT,
    feature_names TEXT
);
