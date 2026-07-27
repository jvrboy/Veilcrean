# How Veilcrean Self-Improves

> The most important architectural choice in Veilcrean is that **the
> confluence engine is not rule-based.** The 8 analysis tools produce
> raw signals, but the *combination* of those signals is learned.

## The feedback loop

```
                ┌──────────────────────────────┐
                │                              │
                │     Live market data         │
                │                              │
                └─────────────┬────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │   8 analysis tools           │
                │   (each emits a score)       │
                └─────────────┬────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │   Feature vector (60-100)    │
                │   ConfluenceEngine output    │
                └─────────────┬────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │   3 Neural Networks          │
                │   A: trade decision          │
                │   B: risk (SL/TP/lot)        │
                │   C: market regime           │
                └─────────────┬────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │   Trade command to EA        │
                └─────────────┬────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │   Trade result (pnl, r)      │
                │   → written to journal       │
                └─────────────┬────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │   Retrainer                  │
                │   every N closed trades      │
                │   ── train on history        │
                │   ── validate on holdout     │
                │   ── deploy if better        │
                └──────────────────────────────┘
```

## What gets logged

Every trade writes one row to `data/trade_journal.db` with:
- `feature_vec` — the **exact** feature vector that produced the decision
- `direction` — BUY / SELL
- `entry_price`, `sl`, `tp`, `lots`
- `confidence` — Network A's confidence at entry
- `regime` — Network C's regime classification
- `session` — Asian / London / NY / Overlap
- `weekday` — 0-6
- `pnl`, `pnl_pct`, `r_achieved`
- `mae`, `mfe` — excursion analytics
- `is_win` — 1/0

So when the retrainer runs, each row becomes one labeled training
sample: `(feature_vec) → (action, confidence, sl, tp, lot, regime)`.

## When retraining happens

Two triggers, configurable in `config.py`:
1. **Every N closed trades** (default 50)
2. **After enough data** (default 100 minimum, then every 50)

## What the retrainer does

1. Read all closed trades from the journal
2. Build training matrices:
   - `X` (N × F) features
   - `y_action` (N,) one of {BUY=0, SELL=1, HOLD=2}
   - `y_conf` (N,) confidence target
   - `y_sl`, `y_tp`, `y_lot` — risk targets normalized to [0, 1]
   - `y_regime` (N,) regime class
3. 80/20 train/holdout split
4. Warm-start from the current deployed weights
5. Train each network for N epochs
6. Validate on holdout:
   - Trade net: needs > 50% accuracy
   - Regime net: needs > 50% accuracy
   - Combined: needs to exceed the threshold (`min_performance_to_deploy`)
7. If new > old → save a new versioned bundle and load it
8. Else → discard the new weights, keep the old model

## Why the safety check?

A new model *always* has the temptation to overfit. The holdout
validation guards against deploying a model that memorized the
training trades but won't generalize.

## Dynamic threshold

In addition to retraining, the **confidence threshold** is adjusted
every cycle based on a rolling 30-trade win rate:
- Recent WR < 40% → threshold up by 0.05 (be pickier)
- Recent WR > 60% → threshold down by 0.02 (take more trades)
- Otherwise    → unchanged

This is a *light* feedback mechanism on top of the neural networks. It
complements the networks rather than fighting them.

## Model versioning

Every successful deployment writes a new file:
```
models/bundle_v20251015_224301.pt
```

You can always roll back:
```python
from python_brain.neural_network import ModelManager
mm = ModelManager()
mm.load_version("v20251015_224301")
```

The brain's next start-up will auto-load the latest bundle.

## Expected timeline

| Stage | Approximate timeline |
|-------|----------------------|
| First 100 trades | Network decisions are basically random — paper trade |
| 100-500 trades | Networks start to learn; performance fluctuates |
| 500-2000 trades | Clear pattern emerges; deploy with caution |
| 2000+ trades | Self-improvement cycle stable; monitor weekly |

> *You should expect 1-3 months of demo trading before going live.*
