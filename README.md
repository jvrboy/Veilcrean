# 🤖 Veilcrean — Adaptive AI Trading Bot

> An institutional-grade, self-learning, multi-timeframe AI trading system that combines classical technical analysis, ICT-style price action, and deep learning to make and execute trading decisions on MetaTrader 5.

---

## 🌌 What is Veilcrean?

**Veilcrean** is a fully autonomous trading system split into two coordinated halves:

| Half | Role | Tech |
|------|------|------|
| **MT5 Expert Advisor** (the *muscle*) | Collects market data, executes trades, manages positions | MQL5 |
| **Python Brain** (the *mind*) | Analyzes, learns, decides, and self-improves | Python 3.11+ / PyTorch |

A ZeroMQ socket links them. The EA is *dumb* — it does no analysis. The Python Brain is *smart* — it does nothing but think, and instructs the EA on what to do.

---

## 🏗️ Architecture at a Glance

```
┌──────────┐    ZMQ    ┌──────────────────────┐
│  MT5 EA  │ ◄──────► │     Python Brain     │
│  (MQL5)  │  JSON     │  • 8 Analysis Tools  │
│  Muscle  │ packets   │  • Confluence Engine │
└──────────┘           │  • 3 Neural Networks │
                       │  • Self-Improver     │
                       └──────────────────────┘
```

### The Python Brain has 5 main modules

1. **Data Receiver & Preprocessor** — JSON → DataFrame, cleans, normalizes, buffers
2. **Analysis Engine** — 8 specialized tools each emitting a score
3. **Confluence Engine** — Combines all scores into a feature vector
4. **Neural Network Engine** — 3 networks decide direction, risk, and regime
5. **Self-Improvement Loop** — Journals every trade, retrains, adapts thresholds

---

## 📁 Project Layout

```
Veilcrean/
├── mt5_ea/                  # MetaTrader 5 Expert Advisor
├── python_brain/            # The mind
│   ├── communication/       # ZMQ server + data parser
│   ├── preprocessor/        # Cleaner, normalizer, buffer
│   ├── analysis_tools/      # 8 analysis tools
│   ├── confluence/          # Feature vector builder
│   ├── neural_network/      # PyTorch models
│   ├── self_improvement/    # Journaling + retraining
│   ├── risk_management/     # Hard safety controls
│   ├── database/            # SQLite journal
│   └── utils/               # Logger, alerts, visualizer
├── models/                  # Saved NN weights
├── data/                    # Journal DB + historical data
├── logs/                    # Rotating log files
├── tests/                   # Unit + integration tests
├── scripts/                 # Maintenance / utility scripts
└── docs/                    # Architecture & strategy docs
```

---

## 🚀 Quick Start

### 1. Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install the MT5 EA

1. Open MetaTrader 5 → `File → Open Data Folder`
2. Copy the contents of `mt5_ea/` into `MQL5/Experts/Veilcrean/`
3. In MetaEditor, compile `Veilcrean_EA.mq5`
4. Drag onto a chart, enable `Allow Algo Trading`
5. Make sure the ZMQ DLL is configured (see `docs/MQ5_ZMQ_SETUP.md`)

### 3. Run the Python Brain

```bash
python -m python_brain.main
```

---

## 🛡️ Safety First

Veilcrean enforces **hard limits** that the neural networks can never override:

- Max daily loss: **3%** of account
- Max total drawdown: **10%** → kill switch
- Max risk per trade: **2%**
- Max open positions: **3**
- Heartbeat timeout: **5s** → flatten all if Python dies
- No trading around high-impact news

---

## ⚡ Design Principles

1. **EA is dumb.** It only collects and executes.
2. **Python is smart.** All decisions live here.
3. **Not rule-based.** Networks learn *which confluences matter*.
4. **Always learning.** Every trade = training data.
5. **Safety first.** Hard limits can never be overridden.
6. **Modular.** Each tool runs and is tested independently.
7. **Versioned.** Every model save is rollback-able.
8. **Logged.** Every decision is fully traceable.

---

## 📅 Build Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation — EA + ZMQ + Python receiver | ✅ |
| 2 | Analysis Tools — all 8 | ✅ |
| 3 | Neural Networks — 3 networks | ✅ |
| 4 | Trade Execution loop | ✅ |
| 5 | Self-Improvement cycle | ✅ |
| 6 | Testing & hardening | ⏳ |
| 7 | Live deployment | ⏳ |

---

## 📚 Documentation

- `docs/ARCHITECTURE.md` — System architecture deep dive
- `docs/STRATEGIES.md` — All supported strategies
- `docs/MQ5_ZMQ_SETUP.md` — ZMQ DLL installation
- `docs/SELF_IMPROVEMENT.md` — How the bot learns over time
- `docs/SAFETY.md` — Risk controls reference

---

## ⚖️ Disclaimer

This software is provided for **educational and research purposes only**. Trading carries substantial risk. Past performance of any strategy — human or AI — is not indicative of future results. Always test on a demo account first. The authors accept no responsibility for financial loss.

---

> *Veilcrean — the unseen mind behind every disciplined trade.*
