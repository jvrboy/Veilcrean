# Veilcrean — System Architecture

> A complete walkthrough of how the two halves of Veilcrean work together.

## The Two Halves

```
┌──────────────────────┐                          ┌──────────────────────────────┐
│  MT5 EA (MQL5)       │  ◄──── ZeroMQ JSON ───► │  Python Brain                │
│  ────                │                          │  ────                        │
│  Data collector      │  pub 5555 ──►            │  ZMQ subscriber              │
│  Trade executor      │  ◄── pull 5556           │  Data parser                 │
│  Position manager    │                          │  Preprocessor                │
│  Heartbeat sender    │  status 5557 ──►         │  8 analysis tools            │
│                      │                          │  Confluence feature builder  │
│  DUMB. No analysis.  │                          │  3 PyTorch networks          │
└──────────────────────┘                          │  Trade journal               │
                                                   │  Retraining loop             │
                                                   │  Risk management             │
                                                   │  Alerts                      │
                                                   │  ────                        │
                                                   │  SMART. All decisions here.  │
                                                   └──────────────────────────────┘
```

### What the EA does
1. Collects candles for **9 timeframes** (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
2. Streams live tick data (bid/ask/spread/volume)
3. Streams account info + open positions
4. Sends a **heartbeat** every 5 seconds
5. Receives trade commands and executes them with slippage protection
6. Manages open positions (trailing stop, partial close, emergency flatten)

### What the Python brain does
1. Receives packets over ZMQ, parses them into typed DTOs
2. Maintains a **rolling buffer** of cleaned, normalized candles per TF
3. Runs **8 analysis tools** each emitting a score in [-1, 1] and a feature dict
4. Builds a **feature vector** of 60-100 numbers
5. Passes the vector through **3 neural networks** (decision, risk, regime)
6. Applies **hard safety checks** (drawdown, daily loss, spread, exposure)
7. Sends trade commands back to the EA
8. Logs every trade in a **SQLite journal**
9. **Retrains** the networks on accumulated data when enough trades exist
10. Adjusts the confidence threshold dynamically based on recent win-rate

## The 8 Analysis Tools

| # | Tool | What it sees |
|---|------|--------------|
| 1 | Market Structure | HH/HL/LH/LL, BOS, CHoCH across all TFs |
| 2 | Supply & Demand | Order blocks, FVGs, nearest S/D zone |
| 3 | Liquidity | Equal highs/lows, stop hunts, liquidity voids |
| 4 | Momentum & Volume | RSI, MACD, divergence, volume z-score |
| 5 | Key Levels | Auto S/R, fib retracements, round numbers |
| 6 | Session & Time | Asian/London/NY, kill zones, day of week |
| 7 | Candlestick Patterns | Engulfing, pin bar, doji, morning star, inside bar |
| 8 | MTF Alignment | % of timeframes agreeing on direction |

Each tool emits a `ToolResult` with:
- `score` in [-1, 1]
- `confidence` in [0, 1]
- `features` — dict of numeric features fed directly to the NN
- `metadata` — human-readable diagnostics (logged, not fed to NN)
- `errors` — any issues encountered

## The 3 Neural Networks

### Network A — Trade Decision
- **Input**: feature vector (size 60-100, dynamic)
- **Output**: 3 logits (BUY, SELL, HOLD) + 1 confidence value
- **Architecture**: Dense(256) → BN → GELU → Dropout → Dense(128) → ... → Dense(64) → heads

### Network B — Risk Management
- **Input**: features + Network A's one-hot action + confidence
- **Output**: SL distance, TP distance, lot size multiplier — all in [0, 1], rescaled at inference
- **Architecture**: Dense(128) → Dense(64) → 3 sigmoid heads

### Network C — Market Regime
- **Input**: features
- **Output**: 5-way logits over (TRENDING, RANGING, VOLATILE, CHOPPY, BREAKOUT)
- **Architecture**: Dense(128) → Dense(64) → Linear(5)

## Safety Systems

**Hard limits** (NN can never override):
- Max daily loss: 3% of account
- Max total drawdown: 10% → kill switch
- Max risk per trade: 2%
- Max open positions: 3
- No trading 30 min around high-impact news
- Heartbeat timeout: 15 s → flatten all

**Soft limits** (NN may adjust within range):
- Confidence threshold: 0.65 – 0.95
- SL: 10 – 100 pips
- TP: 20 – 300 pips
- Lot size: 0.01 – 5.00

## Self-Improvement Cycle

```
   trade opens
       │
       ▼
   trade closes ──► journal entry with feature_vec, regime, pnl, r_achieved
       │
       ▼
   every N closed trades:
       │
       ├─► build training matrices
       ├─► train/val split (80/20)
       ├─► train 3 networks
       ├─► validate on holdout
       ├─► if new > old: deploy + save version
       └─► else: keep old model (safety)

   every cycle:
       └─► update dynamic confidence threshold
           (raise if recent WR < 40%, lower if > 60%)
```

## Data Flow Per Tick

```
   MT5 tick
      │
      ▼
   EA serializes JSON packet
      │
      ▼
   Python ZMQServer.receive_market_data()
      │
      ▼
   DataParser.parse() → MarketSnapshot
      │
      ▼
   BufferManager.update()  (clean, normalize)
      │
      ▼
   ConfluenceEngine.run()
      │   ├─► 8 analysis tools in parallel
      │   ├─► FeatureBuilder.build() → 1D vector
      │   └─► aggregate score
      │
      ▼
   DecisionEngine.decide()
      │   ├─► Net A: action + confidence
      │   ├─► Net C: regime
      │   └─► Net B: sl, tp, lot
      │
      ▼
   Risk checks (drawdown, spread, exposure, heartbeat)
      │
      ▼
   if all pass and confidence > dynamic threshold:
      │   build TRADE_COMMAND
      │   send via ZMQ
      ▼
   EA receives → opens position → sends EXEC_RESULT
      │
      ▼
   Journal entry created
```

## File Map

| Path | Purpose |
|------|---------|
| `mt5_ea/Veilcrean_EA.mq5` | Main EA — wires everything together |
| `mt5_ea/DataCollector.mqh` | OHLCV snapshot builder for 9 TFs |
| `mt5_ea/TradeExecutor.mqh` | Order ops, partial close, flatten, trailing |
| `mt5_ea/SocketLib.mqh` | ZMQ pub/sub wrapper |
| `mt5_ea/Heartbeat.mqh` | Liveness payload |
| `python_brain/main.py` | Orchestrator + main loop |
| `python_brain/communication/` | ZMQ server + JSON parser |
| `python_brain/preprocessor/` | Cleaner, normalizer, buffer |
| `python_brain/analysis_tools/` | 8 analysis tools |
| `python_brain/confluence/` | Feature builder |
| `python_brain/neural_network/` | 3 PyTorch nets, trainer, validator, manager |
| `python_brain/self_improvement/` | Journal, retrainer, perf tracker, threshold |
| `python_brain/risk_management/` | Sizer, drawdown guard, exposure manager |
| `python_brain/database/` | SQLite manager + migrations |
| `python_brain/utils/` | Logger, alerts, visualizer |
