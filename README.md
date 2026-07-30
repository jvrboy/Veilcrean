## 🚀 Apex Evolution (V3.1)

Veilcrean has been fortified into an institutional-grade trading ecosystem:
- **153 Analysis Tools**: 150 technical tools (ICT/SMC, John Ehlers' DSP filters, geometric and statistical measures) + 3 NeuroSense cognitive tools (semantic reasoning, pattern memory, adaptive thresholds).
- **55 Neural Networks**: A hierarchical ensemble (Transformers, Mamba SSM, GNNs, LSTMs, Neural ODEs, Bayesian Nets, and Quantum Simulators).
- **NeuroSense v1.0.0 Cognitive Engine**: A fully self-contained cognitive architecture (eyes, ears, brain, neurons, knowledge graph, learning, language) powering the new cognitive analysis tools.
- **Direct Broker Integration**: Full WebSocket support for **Deriv API** (24/7 perpetual markets) and ZMQ for **MT5**.
- **Cross-Platform Mastery**: One-click setup for Windows, Linux, Docker, and Google Colab.
- **MCP Server (15 Tools)**: Native Model Context Protocol server exposing analysis, cognitive reasoning, pattern recall, adaptive thresholds, brain cognition, and classifier training to external AI agents.
- **Multi-Agent System (MAS)**: 15 specialized agents orchestrated by a super-agent coordinator.


> An institutional-grade, self-learning, multi-timeframe AI trading system that combines classical technical analysis, ICT-style price action, deep learning, and cognitive reasoning to make and execute trading decisions on MetaTrader 5.

---

## 🌌 What is Veilcrean?

**Veilcrean** is a fully autonomous trading system split into two coordinated halves:

| Half | Role | Tech |
|------|------|------|
| **MT5 Expert Advisor** (the *muscle*) | Collects market data, executes trades, manages positions | MQL5 |
| **Python Brain** (the *mind*) | Analyzes, learns, decides, and self-improves | Python 3.11+ / PyTorch |

A ZeroMQ socket links them. The EA is *dumb* — it does no analysis. The Python Brain is *smart* — it does nothing but think, and instructs the EA on what to do.

---

## 🧠 NeuroSense Cognitive Engine (v1.0.0)

Veilcrean now includes **NeuroSense** — an original, fully self-contained cognitive architecture for Python. No AI API providers, no pretrained models, no emotions. Everything is computed locally from first principles using only `numpy`.

| Module | Biological Analogue | What It Does |
|---|---|---|
| `neurosense.eyes` | Retina + visual cortex | Edge detection (Sobel), Harris corners, blob detection, Gaussian blur, image signatures, one-shot recognition |
| `neurosense.ears` | Cochlea + auditory cortex | FFT spectra, spectrograms, mel filterbanks, pitch detection, onset detection, note naming, WAV loading |
| `neurosense.neurons` | Neurons + synapses | Backprop networks, Hebbian learning (Oja's rule), Hopfield associative memory, spiking (LIF) networks with STDP |
| `neurosense.brain` | Prefrontal cortex + hippocampus | Working memory, episodic memory, attention (novelty + habituation), Brain orchestrator |
| `neurosense.knowledge` | Semantic memory | Knowledge graph with confidence, spreading activation, path finding, forward-chaining inference engine |
| `neurosense.learning` | Basal ganglia | Tabular Q-learning, KMeans (k-means++), Self-Organizing Maps |
| `neurosense.language` | Language cortex | TF-IDF similarity, co-occurrence association, n-gram generation, fact extraction from English |

### New Cognitive Analysis Tools

1. **CognitiveReasonerTool** — Translates technical scores into English facts, feeds them to the knowledge graph, and uses the inference engine to derive conclusions about market behavior.
2. **PatternMemoryTool** — Records completed trades as episodic memories and recalls similar past setups to score the current one by historical win rate.
3. **AdaptiveThresholdTool** — Uses Q-learning to discover the optimal confidence threshold per market regime, learning from trade outcomes.

---

## 🏗️ Architecture at a Glance

```
┌──────────┐    ZMQ    ┌──────────────────────┐
│  MT5 EA  │ ◄──────► │     Python Brain     │
│  (MQL5)  │  JSON     │  • 153 Analysis Tools│
│  Muscle  │ packets   │  • Confluence Engine │
└──────────┘           │  • 3 Neural Networks │
                       │  • NeuroSense Brain  │
                       │  • Self-Improver     │
                       └──────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   MCP Server     │
                    │   15 Tools       │
                    │  (stdio JSON-RPC) │
                    └──────────────────┘
```

### The Python Brain has 6 main modules

1. **Data Receiver & Preprocessor** — JSON → DataFrame, cleans, normalizes, buffers
2. **Analysis Engine** — 153 specialized tools each emitting a score
3. **Confluence Engine** — Combines all scores into a feature vector
4. **Neural Network Engine** — 3 networks decide direction, risk, and regime
5. **NeuroSense Cognitive Engine** — Semantic reasoning, pattern memory, adaptive thresholds
6. **Self-Improvement Loop** — Journals every trade, retrains, adapts thresholds

---

## 📁 Project Layout

```
Veilcrean/
├── mt5_ea/                  # MetaTrader 5 Expert Advisor
├── python_brain/            # The mind
│   ├── communication/       # ZMQ server + data parser
│   ├── preprocessor/        # Cleaner, normalizer, buffer
│   ├── analysis_tools/      # 153 analysis tools (150 technical + 3 cognitive)
│   ├── confluence/          # Feature vector builder
│   ├── neural_network/      # PyTorch models
│   ├── self_improvement/    # Journaling + retraining
│   ├── risk_management/     # Hard safety controls
│   ├── database/            # SQLite journal
│   └── utils/               # Logger, alerts, visualizer
├── neurosense/              # Cognitive architecture (v1.0.0)
│   ├── neurosense/
│   │   ├── eyes/            # Visual perception
│   │   ├── ears/            # Auditory perception
│   │   ├── neurons/         # Neural networks from scratch
│   │   ├── brain/           # Memory, attention, orchestration
│   │   ├── knowledge/       # Knowledge graph + inference
│   │   ├── learning/        # Q-learning, KMeans, SOM
│   │   └── language/        # NLP, fact extraction, generation
│   ├── tests/               # 32-test cognitive suite
│   └── examples/            # 4 demo scripts
├── mcp_server.py            # MCP server (15 tools)
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

### 2. Install NeuroSense

```bash
cd neurosense
pip install .
```

### 3. Run NeuroSense Tests

```bash
cd neurosense
python tests/test_all.py
```

### 4. Install the MT5 EA

1. Open MetaTrader 5 → `File → Open Data Folder`
2. Copy the contents of `mt5_ea/` into `MQL5/Experts/Veilcrean/`
3. In MetaEditor, compile `Veilcrean_EA.mq5`
4. Drag onto a chart, enable `Allow Algo Trading`
5. Make sure the ZMQ DLL is configured (see `docs/MQ5_ZMQ_SETUP.md`)

### 5. Run the Python Brain

```bash
python -m python_brain.main
```

### 6. Run the MCP Server

```bash
python mcp_server.py
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
9. **Cognitive.** NeuroSense adds semantic reasoning and memory to the analysis.

---

## 📅 Build Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation — EA + ZMQ + Python receiver | ✅ |
| 2 | Analysis Tools — all 150 technical | ✅ |
| 3 | Neural Networks — 3 networks | ✅ |
| 4 | Trade Execution loop | ✅ |
| 5 | Self-Improvement cycle | ✅ |
| 6 | NeuroSense cognitive engine (v1.0.0) | ✅ |
| 7 | 3 cognitive analysis tools | ✅ |
| 8 | MCP server (15 tools) | ✅ |
| 9 | Testing & hardening | ⏳ |
| 10 | Live deployment | ⏳ |

---

## 📚 Documentation

- `docs/ARCHITECTURE.md` — System architecture deep dive
- `docs/STRATEGIES.md` — All supported strategies
- `docs/MQ5_ZMQ_SETUP.md` — ZMQ DLL installation
- `docs/SELF_IMPROVEMENT.md` — How the bot learns over time
- `docs/SAFETY.md` — Risk controls reference
- `neurosense/README.md` — NeuroSense cognitive architecture docs

---

## ⚖️ Disclaimer

This software is provided for **educational and research purposes only**. Trading carries substantial risk. Past performance of any strategy — human or AI — is not indicative of future results. Always test on a demo account first. The authors accept no responsibility for financial loss.

---

> *Veilcrean — the unseen mind behind every disciplined trade.*
