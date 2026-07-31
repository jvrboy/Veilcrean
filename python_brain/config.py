"""
config.py
=========
Central configuration for the Veilcrean Python brain.

All runtime tunables live here so we never have magic numbers scattered
through the codebase. Edit this file (or override via environment vars)
to retune the bot.

Sections:
    1. Paths
    2. ZMQ communication
    3. Data handling
    4. Analysis tools
    5. Neural networks
    6. Self-improvement
    7. Risk management (hard safety)
    8. Alerts
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean environment variables for production deployments."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR:    Path = PROJECT_ROOT / "data"
MODELS_DIR:  Path = PROJECT_ROOT / "models"
LOGS_DIR:    Path = PROJECT_ROOT / "logs"

JOURNAL_DB:  Path = DATA_DIR / "trade_journal.db"
HISTORICAL:  Path = DATA_DIR / "historical"
BACKTESTS:   Path = DATA_DIR / "backtest_results"

for p in (DATA_DIR, MODELS_DIR, LOGS_DIR, HISTORICAL, BACKTESTS):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. ZMQ communication
# ---------------------------------------------------------------------------
@dataclass
class ZMQConfig:
    """End-points the Python brain listens on / pushes to."""
    # EA publishes market data here; Python subscribes.
    market_data_endpoint:  str = os.getenv("VEIL_ZMQ_PUB", "tcp://127.0.0.1:5555")
    # Python pushes trade commands here; EA pulls.
    trade_command_endpoint: str = os.getenv("VEIL_ZMQ_PULL", "tcp://127.0.0.1:5556")
    # Optional: dashboard can subscribe to brain status
    brain_status_endpoint:  str = os.getenv("VEIL_ZMQ_STATUS", "tcp://127.0.0.1:5557")
    recv_timeout_ms: int = _env_int("VEIL_ZMQ_RECV_TIMEOUT_MS", 1000)
    send_timeout_ms: int = _env_int("VEIL_ZMQ_SEND_TIMEOUT_MS", 1000)

# ---------------------------------------------------------------------------
# 3. Data handling
# ---------------------------------------------------------------------------
TIMEFRAMES: Tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
CANDLE_HISTORY: int = 500        # candles per TF kept in rolling buffer

# ---------------------------------------------------------------------------
# 4. Analysis tools
# ---------------------------------------------------------------------------
@dataclass
class AnalysisConfig:
    """Per-tool tunables."""
    # Market structure
    swing_lookback: int = 5
    bos_min_lookback: int = 10
    # Supply & demand
    ob_min_impulse_pips: float = 10.0
    fvg_min_pips: float = 3.0
    # Liquidity
    eqh_eql_tolerance_pips: float = 2.0
    # Momentum
    rsi_period: int = 14
    macd_fast:   int = 12
    macd_slow:   int = 26
    macd_signal: int = 9
    # Sessions (UTC hours)
    asian_start:  int = 0
    asian_end:    int = 8
    london_start: int = 8
    london_end:   int = 16
    ny_start:     int = 13
    ny_end:       int = 22

# ---------------------------------------------------------------------------
# 5. Neural networks
# ---------------------------------------------------------------------------
@dataclass
class NNConfig:
    """Architecture & training parameters for the three networks."""
    input_dim: int = 64         # set dynamically by feature builder
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    train_test_split: float = 0.8
    # Decision network outputs: BUY, SELL, HOLD
    n_actions: int = 3
    # Regime classes: TRENDING, RANGING, VOLATILE, CHOPPY, BREAKOUT
    n_regimes: int = 5
    # Retraining cadence
    retrain_every_n_trades: int = 50
    retrain_min_samples:    int = 100
    holdout_fraction:       float = 0.2

# ---------------------------------------------------------------------------
# 6. Self-improvement
# ---------------------------------------------------------------------------
@dataclass
class SelfImprovementConfig:
    """Learning loop controls."""
    retrain_every_n_trades: int = 50
    min_trades_for_retrain: int = 100
    confidence_threshold: float = 0.65      # soft floor
    confidence_threshold_max: float = 0.95  # soft ceiling
    min_performance_to_deploy: float = 0.52  # new model must beat this on holdout
    model_version_prefix: str = "v"

# ---------------------------------------------------------------------------
# 7. Risk management — HARD LIMITS, NN can never override
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    """Hard safety controls."""
    # Per trade
    max_risk_per_trade_pct: float = 2.0
    # Per day
    max_daily_loss_pct: float = 3.0
    # Overall
    max_total_drawdown_pct: float = 10.0     # hit → KILL SWITCH
    # Portfolio
    max_open_positions: int = 3
    max_correlated_positions: int = 1        # same direction on same pair family
    # Execution
    max_spread_points: float = 30.0
    news_buffer_minutes: int = 30
    flatten_friday_hour: int = 16            # 16:00 server time Friday
    heartbeat_timeout_sec: int = 15          # if no heartbeat → flatten all

    # --- Soft limits (NN may adjust within range) ---
    confidence_min: float = 0.65
    confidence_max: float = 0.95
    sl_min_pips: float = 10.0
    sl_max_pips: float = 100.0
    tp_min_pips: float = 20.0
    tp_max_pips: float = 300.0
    lot_min:     float = 0.01
    lot_max:     float = 5.0

# ---------------------------------------------------------------------------
# 8. Alerts
# ---------------------------------------------------------------------------
@dataclass
class AlertConfig:
    """Notification endpoints (optional)."""
    telegram_bot_token: str = os.getenv("VEIL_TG_TOKEN", "")
    telegram_chat_id:   str = os.getenv("VEIL_TG_CHAT", "")
    discord_webhook_url: str = os.getenv("VEIL_DISCORD_HOOK", "")
    enable_console: bool = True
    # Which events to push
    notify_on_trade_open:   bool = True
    notify_on_trade_close:  bool = True
    notify_on_kill_switch:  bool = True
    notify_on_retrain:      bool = True
    notify_on_errors:       bool = True

# ---------------------------------------------------------------------------
# 9. LLM / AI Reasoning
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Settings for Gemini and Groq reasoning."""
    provider: str = os.getenv("VEIL_LLM_PROVIDER", "groq") # 'groq' or 'gemini'
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    # Models
    groq_model: str = "llama3-70b-8192"
    gemini_model: str = "gemini-1.5-pro"
    
    # Cadence
    reason_every_n_ticks: int = _env_int("VEIL_LLM_REASON_EVERY_N_TICKS", 50) # LLMs are slow/expensive, don't run every tick
    enabled: bool = _env_bool("VEIL_LLM_ENABLED", False) # Disabled by default unless explicitly enabled

# ---------------------------------------------------------------------------
# 10. Deriv API Configuration
# ---------------------------------------------------------------------------
@dataclass
class DerivConfig:
    """Settings for trading directly on Deriv."""
    app_id: int = _env_int("DERIV_APP_ID", 0)
    api_token: str = os.getenv("DERIV_API_TOKEN", "")
    is_demo: bool = _env_bool("DERIV_IS_DEMO", True)
    enabled: bool = _env_bool("DERIV_ENABLED", False)

# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
ZMQ_CFG  = ZMQConfig()
ANA_CFG  = AnalysisConfig()
NN_CFG   = NNConfig()
SI_CFG   = SelfImprovementConfig()
RISK_CFG = RiskConfig()
ALERT_CFG = AlertConfig()
LLM_CFG  = LLMConfig()
DERIV_CFG = DerivConfig()
