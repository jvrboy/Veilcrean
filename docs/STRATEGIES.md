# Supported Strategies & Concepts

Veilcrean is not a single-strategy EA. The 8 analysis tools collectively
implement and reason about the following strategies and concepts. The
neural networks learn *which combinations* work best in which regime —
not hard-coded rules.

---

## 1. Trend-Following
- Moving Average Crossovers (Golden/Death Cross)
- MACD signal line crosses
- ADX strength filtering
- Parabolic SAR flips
- Ichimoku Cloud breakouts
- Supertrend flips
- Higher-highs / higher-lows sequences
- EMA-50 slope (used by the MTF Alignment tool)

## 2. Mean Reversion
- Bollinger Band bounces & squeezes
- RSI overbought/oversold
- Stochastic Oscillator extremes
- CCI extremes
- Keltner Channel re-entries

## 3. Momentum
- RSI divergence (price vs. RSI)
- MACD histogram divergence
- Rate of Change (ROC)
- Williams %R
- Money Flow Index (MFI)

## 4. Volume-Based
- On-Balance Volume (OBV)
- VWAP (approximated)
- Accumulation/Distribution Line
- Chaikin Money Flow
- Volume z-score (used by the Momentum tool)

## 5. Breakout / Breakdown
- Support & Resistance breakouts
- Donchian Channel breakouts
- Volatility breakouts (ATR-based)
- Opening Range Breakouts (ORB)
- 52-week high/low breakouts
- Liquidity voids → displacement

## 6. Chart Patterns
- Head and Shoulders / Inverse
- Double Top / Double Bottom
- Cup and Handle
- Ascending / Descending / Symmetrical Triangles
- Bull / Bear Flags & Pennants
- Rising / Falling Wedges
- Rounding Bottoms

## 7. Candlestick Patterns
- Engulfing (Bullish / Bearish)
- Doji reversals
- Hammer / Inverted Hammer
- Shooting Star
- Morning Star / Evening Star
- Three White Soldiers / Three Black Crows
- Harami
- Tweezer Tops / Bottoms
- Marubozu
- Pin Bar
- Inside / Outside Bar

## 8. Fibonacci
- Retracement (23.6 / 38.2 / 50 / 61.8 / 78.6)
- Extension (127.2 / 161.8)
- Fibonacci fans & time zones (planner)

## 9. Elliott Wave / Harmonic (planned)
- Elliott Wave impulse / corrective labelling
- Gartley, Bat, Butterfly, Crab
- Wolfe Wave

## 10. ICT / Smart Money Concepts
This is where Veilcrean shines:
- **Break of Structure (BOS)** — tool 1
- **Change of Character (CHoCH)** — tool 1
- **Order Blocks** — tool 2
- **Fair Value Gaps (FVG)** — tool 2
- **Supply & Demand zones** — tool 2
- **Equal Highs / Equal Lows** — tool 3
- **Liquidity Sweeps / Stop Hunts** — tool 3
- **Optimal Trade Entry (OTE)** — tool 5 (via fib)
- **Displacement** — tool 3 (volatility void)
- **Mitigation / Breaker / Rejection blocks** — combined tools 1+2
- **Premium / Discount zones** — implicit in tool 5

## 11. Multi-Timeframe
- Top-down analysis (HTF bias + LTF entry)
- Triple Screen (Elder)
- TF alignment score (tool 8)

## 12. Session / Time-Based
- Asian range fades
- London breakouts
- NY open momentum
- London/NY overlap (peak liquidity)
- Kill zones (first 1-2 hours of London & NY)
- Day-of-week bias (Tue-Thu stronger)
- Friday afternoon flatten

## 13. Volatility
- ATR-based trailing stops
- Bollinger Bandwidth expansion
- Realized volatility (used by Normalizer)

---

## How the strategies map to the analysis tools

| Strategy family | Primary tool(s) | Score driver |
|-----------------|-----------------|--------------|
| Trend-following | 1, 8 | direction alignment |
| Mean reversion  | 4, 5 | distance from level |
| Momentum        | 4 | RSI / MACD |
| Volume          | 4 | volume z-score |
| Breakout        | 3, 8 | liquidity + alignment |
| Chart pattern   | 1, 5 | structure + levels |
| Candlestick     | 7 | pattern score |
| Fibonacci       | 5 | nearest fib level |
| ICT             | 1, 2, 3, 5 | BOS/OB/FVG/sweep |
| MTF             | 8 | TF alignment % |
| Session         | 6 | time of day |
| Volatility      | 3, 4 | ATR / vol z |

The networks learn, per regime, which subset of these signals is
*actually predictive* of forward returns. The 3-network architecture
ensures we don't just learn the *direction* but also the *size* of
the trade (Network B) and *when* the strategy is appropriate at all
(Network C).
