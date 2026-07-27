"""
test_data_parser.py
===================
Tests for the JSON → MarketSnapshot parser.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from datetime import datetime

import pandas as pd
import pytest

from python_brain.communication.data_parser import DataParser, mid_price


SAMPLE = {
    "type": "MARKET_DATA",
    "symbol": "EURUSD",
    "trigger": "TICK",
    "timestamp": 1700000000,
    "tick": {"bid": 1.0840, "ask": 1.0845, "spread": 1.5, "volume": 250},
    "candles": {
        "M5": [
            {"O": 1.080, "H": 1.081, "L": 1.079, "C": 1.0805, "V": 100, "t": 1700000000},
            {"O": 1.0805, "H": 1.082, "L": 1.080, "C": 1.0815, "V": 120, "t": 1700000300},
        ],
        "H1": [
            {"O": 1.080, "H": 1.090, "L": 1.070, "C": 1.085, "V": 1200, "t": 1700000000},
        ],
    },
    "account": {
        "balance": 10000, "equity": 10100, "margin_free": 9500,
        "margin_used": 500, "profit": 100, "leverage": 100,
    },
    "positions": [
        {"ticket": 12345, "type": "BUY", "symbol": "EURUSD",
         "lots": 0.1, "open": 1.08, "sl": 1.07, "tp": 1.10, "profit": 20.0}
    ]
}


def test_parser_market_data():
    p = DataParser()
    snap = p.parse(SAMPLE)
    assert snap is not None
    assert snap.symbol == "EURUSD"
    assert snap.tick.bid == 1.0840
    assert "M5" in snap.candles
    assert len(snap.candles["M5"]) == 2
    assert snap.account.balance == 10000
    assert len(snap.positions) == 1
    assert snap.positions[0].ticket == 12345
    assert isinstance(snap.timestamp, datetime)


def test_parser_account_update():
    p = DataParser()
    raw = {"type": "ACCOUNT_UPDATE", "symbol": "EURUSD",
           "timestamp": 1700000000, "trigger": "TRADE_EVENT",
           "account": SAMPLE["account"], "positions": []}
    snap = p.parse(raw)
    assert snap is not None
    assert snap.candles == {}
    assert snap.account.balance == 10000


def test_parser_heartbeat_returns_none():
    p = DataParser()
    assert p.parse({"type": "HEARTBEAT"}) is None


def test_mid_price_from_tick():
    p = DataParser()
    snap = p.parse(SAMPLE)
    assert mid_price(snap) == pytest.approx((1.0840 + 1.0845) / 2, rel=1e-6)


def test_mid_price_falls_back_to_close():
    p = DataParser()
    snap = p.parse(SAMPLE)
    snap.tick = None
    assert mid_price(snap) == pytest.approx(1.0815, rel=1e-4)
