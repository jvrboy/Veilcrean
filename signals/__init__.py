"""Veilcrean self-learning signal package.

Modules:
    data_feeds      real market data (yfinance + Deriv WebSocket)
    signal_engine   144-tool confluence -> ENTRY/TP/SL
    learning        per-instrument adaptive parameters (self-improvement)
    tracker         signal ledger + TP/SL performance evaluator
    generate_signals  batch entry point (run this)
"""
