# Installing ZMQ for the MT5 EA

The MT5 EA talks to the Python brain over **ZeroMQ**. MQL5 doesn't have
native ZMQ support, so we link to a DLL.

## Step 1 — Get the libzmq DLL

### Windows (most common for MT5)
1. Download a prebuilt libzmq for Windows. The official ZeroMQ project
   does not ship Windows binaries, but there are excellent community
   builds:
   - https://github.com/zeromq/libzmq/releases
   - The file you want is typically `libzmq-v140-x64-4_3_5.dll`
     (Visual Studio 2015 / x64) or a similar variant for your compiler.

2. Rename the file to **`libzmq.dll`** (or update the `#import` directive
   in `SocketLib.mqh` to point at the exact name you downloaded).

3. Copy `libzmq.dll` to your MT5 terminal's `MQL5/Libraries/` folder:
   ```
   <MT5 Data Folder>\MQL5\Libraries\libzmq.dll
   ```

### macOS / Linux (MT5 via Wine or similar)
- Install via Homebrew: `brew install zeromq`
- Symlink the `.dylib` / `.so` into MT5's `Libraries/` directory.

## Step 2 — Allow DLL imports

In MetaTrader 5:
- `Tools → Options → Expert Advisors`
- Tick **"Allow DLL imports"**
- Optionally tick **"Confirm DLL function calls"** for debugging

## Step 3 — Compile the EA

1. Open MetaEditor (`F4` in MT5).
2. Open `MQL5/Experts/Veilcrean/Veilcrean_EA.mq5`.
3. Compile (`F7`). Expect zero errors / zero warnings.

## Step 4 — Configure endpoints

The EA defaults to:
- `tcp://127.0.0.1:5555` (EA → Python, market data)
- `tcp://127.0.0.1:5556` (Python → EA, trade commands)

If you need to change these, edit the EA's `Inputs` panel after
attaching it to a chart, or set the corresponding env vars for the
Python side:
```bash
export VEIL_ZMQ_PUB="tcp://127.0.0.1:5555"
export VEIL_ZMQ_PULL="tcp://127.0.0.1:5556"
```

## Step 5 — Sanity check

1. Start the Python brain first:
   ```bash
   python -m python_brain.main
   ```
2. Then attach the EA to a chart.
3. You should see the Python brain log incoming `MARKET_DATA` packets
   every tick and the EA log `VEILCREAN: pushed TICK packet`.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `libzmq.dll not found` | DLL not in `Libraries/`, or wrong arch (x86 vs x64) |
| `zmq_ctx_new returns 0` | DLL is incompatible; try a different build |
| `PYTHON: timeout receiving` | Wrong port, or Python not running |
| Heartbeat lost within 5s | Firewall or wrong ZMQ endpoint |
| Trades not opening | Check `Allow Algo Trading` is on, and the magic number matches |

## ZMQ topology recap

```
EA publishes  on  tcp://127.0.0.1:5555   (PUB)  → Python subscribes (SUB)
EA pulls      on  tcp://127.0.0.1:5556   (PULL) ← Python pushes    (PUSH)
Python pubs   on  tcp://127.0.0.1:5557   (PUB)  → dashboard can sub
```

All sockets are local-only by default. If you run Python on a
different machine from MT5, change the IP and open the ports on both
firewalls.
