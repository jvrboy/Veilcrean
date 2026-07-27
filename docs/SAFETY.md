# Safety & Risk Controls

> **The neural networks can never override hard safety limits.** This
> is a non-negotiable design principle of Veilcrean.

## Two-tier risk model

### Tier 1 — Hard limits
These checks run on every decision, before any command is sent. The
NNs can never raise them or skip them.

| Limit | Default | Action on breach |
|-------|---------|------------------|
| Max risk per trade | 2% | Lot size capped |
| Max daily loss | 3% | No new trades |
| Max total drawdown | 10% | **Kill switch** — flatten all |
| Max open positions | 3 | No new trades |
| Max correlated positions | 1 | No new trades |
| Max spread | 30 pts | Trade rejected |
| News buffer | 30 min | No new trades |
| Heartbeat timeout | 15 s | Flatten all |
| Friday flatten | 16:00 server | Force-close all |

### Tier 2 — Soft limits
The neural network chooses within these ranges. The risk management
module clamps the values after the NN but before they reach the EA.

| Limit | Min | Max |
|-------|-----|-----|
| Confidence threshold | 0.65 | 0.95 |
| SL (pips) | 10 | 100 |
| TP (pips) | 20 | 300 |
| Lot size | 0.01 | 5.00 |

## How the kill switch works

```python
if snapshot.account:
    self.guard.update(snapshot.account.equity)
    if self.guard.kill_switch:
        alerter.kill_switch(self.guard.kill_reason)
        # send FLATTEN_ALL to EA
        self.zmq.send_trade_command({"action": "FLATTEN_ALL", "symbol": ...})
```

When the kill switch fires:
1. The brain sends a `FLATTEN_ALL` command to the EA
2. The EA closes every open position on the symbol
3. The brain stops sending new trade commands
4. An alert is fired to Telegram/Discord
5. The kill reason is logged

The kill switch only resets on manual intervention (restart the brain).

## What "2% risk per trade" actually means

```python
risk_dollars = account_balance * 0.02
lots         = risk_dollars / (sl_pips * pip_value_per_lot)
```

So if your account is $10,000, max risk per trade = $200. If your SL
is 20 pips and pip value is $10/lot, max lots = 1.0. The sizer
returns this directly, then it gets clamped by the soft limits.

## Position correlation

Veilcrean maintains a simple group map:

```
EUR-group : EURUSD, EURGBP, EURJPY, EURCHF, EURAUD, EURCAD
GBP-group : GBPUSD, EURGBP, GBPJPY, GBPCHF, GBPAUD
JPY-group : USDJPY, EURJPY, GBPJPY, AUDJPY, NZDJPY, CADJPY
...
```

If you already have a long EURUSD, the bot won't open another long
position on any EUR-group pair until the first closes.

## News filter (planned)

In `RiskConfig.news_buffer_minutes` you can configure a buffer around
high-impact news events. Veilcrean ships with a placeholder — wire
your preferred news source (ForexFactory JSON, Investing.com, etc.)
into the risk check to enable.

## What you should monitor

1. **Daily P&L** — at most -3% of starting equity
2. **Drawdown** — should rarely approach 10%
3. **Win rate per regime** — if TRENDING is losing, your regime classifier is off
4. **Confidence calibration** — when bot says 0.8, it should win 80% of the time
5. **Feature importance drift** — which tools are predictive today vs. last month
