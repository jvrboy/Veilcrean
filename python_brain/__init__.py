"""
Veilcrean Python Brain
======================
The *mind* of the trading system. All analysis, learning, and decision
making happens here. The MT5 EA is the *muscle* — it just collects and
executes. This package is organized into focused subpackages:

    communication    — ZMQ server + data parser
    preprocessor     — Cleaner, normalizer, rolling buffer
    analysis_tools   — 8 specialized analysis tools
    confluence       — Feature vector builder
    neural_network   — 3 PyTorch models + trainer/manager
    self_improvement — Journal, retrainer, performance tracker
    risk_management  — Hard safety controls
    database         — SQLite trade journal
    utils            — Logger, alerts, visualizer
"""
__version__ = "1.0.0"
__author__  = "Veilcrean"
