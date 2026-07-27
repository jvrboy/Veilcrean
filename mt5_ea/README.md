# Veilcrean — MT5 Expert Advisor

This directory contains the MQL5 code for the MT5 side of Veilcrean.

## Files

| File | Purpose |
|------|---------|
| `Veilcrean_EA.mq5` | The main EA — wires the modules together and runs the OnTick loop |
| `DataCollector.mqh` | Collects OHLCV data for all 9 timeframes + tick + account + positions |
| `TradeExecutor.mqh` | Receives trade commands from Python, executes them, manages positions |
| `SocketLib.mqh` | ZeroMQ wrapper (talks to libzmq DLL) |
| `Heartbeat.mqh` | Sends a heartbeat every N seconds so Python knows the EA is alive |

## Installation

See `../docs/MQ5_ZMQ_SETUP.md` for full instructions.

Quick version:
1. Install `libzmq.dll` in `MQL5/Libraries/`
2. Copy these 5 files into `MQL5/Experts/Veilcrean/`
3. Open `Veilcrean_EA.mq5` in MetaEditor, compile
4. Allow algo trading and DLL imports in MT5 options
5. Drag onto a chart, set the inputs (ZMQ address, ports, etc.)

## Input parameters

| Input | Default | Description |
|-------|---------|-------------|
| `InpZmqAddress` | `tcp://127.0.0.1:5555` | Where EA publishes market data |
| `InpZmqPullPort` | `5556` | Port Python pushes commands to |
| `InpHeartbeatSec` | `5` | Heartbeat interval in seconds |
| `InpTickPushMs` | `250` | Min ms between tick pushes (anti-flood) |
| `InpCandleHistory` | `200` | Candles per TF to send in each packet |
| `InpDebugLog` | `true` | Verbose logging |
| `InpMaxSlippage` | `3.0` | Max slippage in points |
| `InpMaxSpreadPts` | `30.0` | Max spread to accept new orders |

## What the EA *does not* do

- No indicator calculations
- No pattern recognition
- No trade decisions
- No SL/TP calculation

It only:
- Collects data
- Sends it to Python
- Receives trade commands
- Executes them with slippage protection
- Manages positions (trailing stop, partial close, flatten)

The Python brain makes every decision.

## Magic number

The EA uses magic number `7772025`. Make sure this doesn't conflict
with any other EA on the same account. To change it, edit
`Veilcrean_EA.mq5` and search for `g_magic`.
