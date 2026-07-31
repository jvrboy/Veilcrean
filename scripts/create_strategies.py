#!/usr/bin/env python3
"""Create 10 tailored strategies per instrument (80 total for 8 instruments)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STRAT_DIR = REPO / "training" / "strategies"
STRAT_DIR.mkdir(parents=True, exist_ok=True)

# Each strategy defines: indicator emphasis weights, regime filters, TP/SL multipliers,
# minimum confidence, and instrument-specific adaptations

VOLATILITY_STRATEGIES = [
    {"id":"vol_rsi_bollinger_reversal","name":"RSI Bollinger Reversal","desc":"Enter when RSI is extreme (>70 or <30) and price is at Bollinger Band edge, expecting mean reversion within volatility range.","indicators":{"rsi":2.0,"bollinger":1.8,"stochastic":1.0},"regime_filter":["RANGING","CHOPPY"],"tp_mult":0.8,"sl_mult":0.6,"min_confidence":0.55,"entry_filter":"rsi_extreme_bollinger_touch"},
    {"id":"vol_macd_trend_follow","name":"MACD Trend Follow","desc":"Follow MACD crossovers with ADX confirmation for strong directional moves in volatility indices.","indicators":{"macd":2.0,"dmi_direction":1.5,"adx":1.2},"regime_filter":["TRENDING"],"tp_mult":2.0,"sl_mult":0.8,"min_confidence":0.45,"entry_filter":"macd_cross_above_zero"},
    {"id":"vol_breakout_squeeze","name":"Squeeze Breakout","desc":"Enter on TTM Squeeze release with ATR expansion - volatility indices show strong post-squeeze moves.","indicators":{"ttm_squeeze":2.0,"atr_channel":1.8,"bollinger_width":1.5},"regime_filter":["BREAKOUT","TRENDING"],"tp_mult":2.5,"sl_mult":1.0,"min_confidence":0.50,"entry_filter":"squeeze_fired_atr_expanding"},
    {"id":"vol_hull_momentum","name":"Hull MA Momentum","desc":"Hull MA direction change with momentum and ROC confirmation for clean trend entries.","indicators":{"hull_ma":2.0,"momentum":1.5,"roc":1.2},"regime_filter":["TRENDING","BREAKOUT"],"tp_mult":1.8,"sl_mult":0.7,"min_confidence":0.45,"entry_filter":"hull_reversal_momentum_align"},
    {"id":"vol_ichimoku_trend","name":"Ichimoku Cloud Trend","desc":"Price above/below Ichimoku cloud with Tenkan-Kijun cross confirmation.","indicators":{"ichimoku":2.0,"multi_tf_trend":1.5},"regime_filter":["TRENDING"],"tp_mult":2.0,"sl_mult":0.8,"min_confidence":0.50,"entry_filter":"ichimoku_cloud_break"},
    {"id":"vol_stochastic_divergence","name":"Stochastic Divergence","desc":"Stochastic RSI divergence with Williams %R confirmation for reversal entries.","indicators":{"stochastic_rsi":2.0,"rsi_divergence":1.8,"williams_r":1.2},"regime_filter":["RANGING","CHOPPY","VOLATILE"],"tp_mult":1.0,"sl_mult":0.5,"min_confidence":0.55,"entry_filter":"stoch_rsi_divergence"},
    {"id":"vol_multi_tf_confluence","name":"Multi-TF Confluence","desc":"Align multiple timeframe trends with DMI direction for high-confidence entries.","indicators":{"multi_tf_trend":2.0,"dmi_direction":1.5,"hull_ma":1.0},"regime_filter":["TRENDING","BREAKOUT"],"tp_mult":2.0,"sl_mult":0.7,"min_confidence":0.50,"entry_filter":"all_tf_aligned"},
    {"id":"vol_adx_breakout","name":"ADX Breakout","desc":"ADX rising above 25 with directional indicator confirming breakout direction.","indicators":{"adx":2.0,"dmi_direction":1.8,"atr_channel":1.2},"regime_filter":["BREAKOUT"],"tp_mult":2.5,"sl_mult":0.8,"min_confidence":0.45,"entry_filter":"adx_rising_di_cross"},
    {"id":"vol_keltner_bounce","name":"Keltner Channel Bounce","desc":"Price bouncing off Keltner Channel with CCI confirmation for mean reversion.","indicators":{"keltner":2.0,"cci":1.5,"rsi":1.0},"regime_filter":["RANGING"],"tp_mult":0.8,"sl_mult":0.5,"min_confidence":0.55,"entry_filter":"keltner_touch_cci_extreme"},
    {"id":"vol_chop_avoidance_trend","name":"Chop Avoidance Trend","desc":"Only trade when chop index is LOW (trending market) with momentum confirmation.","indicators":{"chop_index":2.0,"momentum":1.5,"macd":1.2},"regime_filter":["TRENDING","BREAKOUT"],"tp_mult":2.0,"sl_mult":0.8,"min_confidence":0.50,"entry_filter":"low_chop_momentum_confirm"},
]

BOOM_STRATEGIES = [
    {"id":"boom_spike_rsi_oversold","name":"Spike RSI Oversold","desc":"BUY when RSI is deeply oversold before a spike. Boom indices spike upward after consolidation.","indicators":{"rsi":2.0,"bollinger":1.5},"regime_filter":["RANGING","CHOPPY","VOLATILE"],"tp_mult":2.5,"sl_mult":1.2,"min_confidence":0.40,"force_direction":"BUY","entry_filter":"rsi_oversold_consolidation"},
    {"id":"boom_squeeze_fire","name":"Squeeze Fire Buy","desc":"Enter BUY on TTM Squeeze release - boom spikes often follow volatility compression.","indicators":{"ttm_squeeze":2.0,"atr_channel":1.8},"regime_filter":["BREAKOUT","VOLATILE","RANGING"],"tp_mult":3.0,"sl_mult":1.0,"min_confidence":0.45,"force_direction":"BUY","entry_filter":"squeeze_release_buy"},
    {"id":"boom_bollinger_lower","name":"Bollinger Lower Buy","desc":"BUY when price touches lower Bollinger Band in ranging market - classic mean reversion for boom spikes.","indicators":{"bollinger":2.0,"rsi":1.5,"stochastic":1.0},"regime_filter":["RANGING","CHOPPY"],"tp_mult":2.0,"sl_mult":1.0,"min_confidence":0.45,"force_direction":"BUY","entry_filter":"bollinger_lower_touch"},
    {"id":"boom_momentum_surge","name":"Momentum Surge Buy","desc":"BUY on momentum surge after quiet period - boom indices show sharp upward momentum bursts.","indicators":{"momentum":2.0,"roc":1.8,"awesome_osc":1.5},"regime_filter":["TRENDING","BREAKOUT","VOLATILE"],"tp_mult":2.5,"sl_mult":1.2,"min_confidence":0.40,"force_direction":"BUY","entry_filter":"momentum_surge_from_low"},
    {"id":"boom_atr_expansion","name":"ATR Expansion Buy","desc":"BUY when ATR starts expanding after contraction - signals imminent boom spike.","indicators":{"atr_channel":2.0,"bollinger_width":1.8},"regime_filter":["BREAKOUT","VOLATILE"],"tp_mult":3.0,"sl_mult":1.0,"min_confidence":0.45,"force_direction":"BUY","entry_filter":"atr_expanding_from_low"},
    {"id":"boom_stochastic_oversold","name":"Stochastic Oversold Buy","desc":"BUY when stochastic crosses up from oversold - early entry before spike.","indicators":{"stochastic":2.0,"williams_r":1.5,"rsi":1.0},"regime_filter":["RANGING","CHOPPY"],"tp_mult":2.0,"sl_mult":1.0,"min_confidence":0.45,"force_direction":"BUY","entry_filter":"stoch_cross_up_oversold"},
    {"id":"boom_dmi_bullish_cross","name":"DMI Bullish Cross Buy","desc":"BUY when +DI crosses above -DI with rising ADX - confirms upward breakout.","indicators":{"dmi_direction":2.0,"adx":1.5,"hull_ma":1.0},"regime_filter":["TRENDING","BREAKOUT"],"tp_mult":2.5,"sl_mult":1.0,"min_confidence":0.45,"force_direction":"BUY","entry_filter":"plus_di_cross_above_minus"},
    {"id":"boom_supply_demand_buy","name":"Supply/Demand Buy","desc":"BUY at demand zone with volume confirmation - boom spikes originate from demand zones.","indicators":{"supply_demand":2.0,"volume_conf":1.5},"regime_filter":["RANGING","VOLATILE"],"tp_mult":2.5,"sl_mult":1.2,"min_confidence":0.40,"force_direction":"BUY","entry_filter":"demand_zone_touch"},
    {"id":"boom_cci_extreme_buy","name":"CCI Extreme Buy","desc":"BUY when CCI is extremely oversold (-200 or below) - high-probability reversal for boom.","indicators":{"cci":2.0,"rsi":1.5,"bollinger":1.0},"regime_filter":["RANGING","CHOPPY","VOLATILE"],"tp_mult":2.0,"sl_mult":1.0,"min_confidence":0.40,"force_direction":"BUY","entry_filter":"cci_below_neg200"},
    {"id":"boom_ichimoku_buy","name":"Ichimoku Cloud Buy","desc":"BUY when price breaks above Ichimoku cloud - strong trend signal for boom spike.","indicators":{"ichimoku":2.0,"multi_tf_trend":1.5},"regime_filter":["TRENDING","BREAKOUT"],"tp_mult":3.0,"sl_mult":1.2,"min_confidence":0.45,"force_direction":"BUY","entry_filter":"price_above_cloud"},
]

CRASH_STRATEGIES = [
    {"id":"crash_rsi_overbought","name":"RSI Overbought Sell","desc":"SELL when RSI is extremely overbought - crash indices drop sharply from overbought levels.","indicators":{"rsi":2.0,"bollinger":1.5},"regime_filter":["RANGING","CHOPPY","VOLATILE"],"tp_mult":2.5,"sl_mult":1.2,"min_confidence":0.40,"force_direction":"SELL","entry_filter":"rsi_overbought_consolidation"},
    {"id":"crash_squeeze_fire","name":"Squeeze Fire Sell","desc":"SELL on TTM Squeeze release - crash drops often follow volatility compression.","indicators":{"ttm_squeeze":2.0,"atr_channel":1.8},"regime_filter":["BREAKOUT","VOLATILE","RANGING"],"tp_mult":3.0,"sl_mult":1.0,"min_confidence":0.45,"force_direction":"SELL","entry_filter":"squeeze_release_sell"},
    {"id":"crash_bollinger_upper","name":"Bollinger Upper Sell","desc":"SELL when price touches upper Bollinger Band - mean reversion for crash drops.","indicators":{"bollinger":2.0,"rsi":1.5,"stochastic":1.0},"regime_filter":["RANGING","CHOPPY"],"tp_mult":2.0,"sl_mult":1.0,"min_confidence":0.45,"force_direction":"SELL","entry_filter":"bollinger_upper_touch"},
    {"id":"crash_momentum_drop","name":"Momentum Drop Sell","desc":"SELL on negative momentum surge - crash indices show sharp downward momentum bursts.","indicators":{"momentum":2.0,"roc":1.8},"regime_filter":["TRENDING","BREAKOUT","VOLATILE"],"tp_mult":2.5,"sl_mult":1.2,"min_confidence":0.40,"force_direction":"SELL","entry_filter":"momentum_dropping"},
    {"id":"crash_atr_expansion","name":"ATR Expansion Sell","desc":"SELL when ATR starts expanding - signals imminent crash drop.","indicators":{"atr_channel":2.0,"bollinger_width":1.8},"regime_filter":["BREAKOUT","VOLATILE"],"tp_mult":3.0,"sl_mult":1.0,"min_confidence":0.45,"force_direction":"SELL","entry_filter":"atr_expanding_high"},
    {"id":"crash_stochastic_overbought","name":"Stochastic Overbought Sell","desc":"SELL when stochastic crosses down from overbought.","indicators":{"stochastic":2.0,"williams_r":1.5,"rsi":1.0},"regime_filter":["RANGING","CHOPPY"],"tp_mult":2.0,"sl_mult":1.0,"min_confidence":0.45,"force_direction":"SELL","entry_filter":"stoch_cross_down_overbought"},
    {"id":"crash_dmi_bearish_cross","name":"DMI Bearish Cross Sell","desc":"SELL when -DI crosses above +DI with rising ADX.","indicators":{"dmi_direction":2.0,"adx":1.5},"regime_filter":["TRENDING","BREAKOUT"],"tp_mult":2.5,"sl_mult":1.0,"min_confidence":0.45,"force_direction":"SELL","entry_filter":"minus_di_cross_above_plus"},
    {"id":"crash_supply_resistance","name":"Supply Zone Sell","desc":"SELL at supply zone - crash drops originate from supply zones.","indicators":{"supply_demand":2.0},"regime_filter":["RANGING","VOLATILE"],"tp_mult":2.5,"sl_mult":1.2,"min_confidence":0.40,"force_direction":"SELL","entry_filter":"supply_zone_touch"},
    {"id":"crash_cci_extreme_sell","name":"CCI Extreme Sell","desc":"SELL when CCI is extremely overbought (+200 or above).","indicators":{"cci":2.0,"rsi":1.5},"regime_filter":["RANGING","CHOPPY","VOLATILE"],"tp_mult":2.0,"sl_mult":1.0,"min_confidence":0.40,"force_direction":"SELL","entry_filter":"cci_above_pos200"},
    {"id":"crash_ichimoku_sell","name":"Ichimoku Cloud Sell","desc":"SELL when price breaks below Ichimoku cloud.","indicators":{"ichimoku":2.0,"multi_tf_trend":1.5},"regime_filter":["TRENDING","BREAKOUT"],"tp_mult":3.0,"sl_mult":1.2,"min_confidence":0.45,"force_direction":"SELL","entry_filter":"price_below_cloud"},
]

INSTRUMENT_STRATEGY_MAP = {
    "1HZ50V":  {"name":"Volatility_50",  "type":"volatility", "strategies": VOLATILITY_STRATEGIES},
    "1HZ75V":  {"name":"Volatility_75",  "type":"volatility", "strategies": VOLATILITY_STRATEGIES},
    "1HZ100V": {"name":"Volatility_100", "type":"volatility", "strategies": VOLATILITY_STRATEGIES},
    "BOOM500":  {"name":"Boom_500",  "type":"boom",  "strategies": BOOM_STRATEGIES},
    "BOOM900":  {"name":"Boom_900",  "type":"boom",  "strategies": BOOM_STRATEGIES},
    "BOOM1000": {"name":"Boom_1000", "type":"boom",  "strategies": BOOM_STRATEGIES},
    "CRASH500": {"name":"Crash_500", "type":"crash", "strategies": CRASH_STRATEGIES},
    "CRASH900": {"name":"Crash_900", "type":"crash", "strategies": CRASH_STRATEGIES},
}

# Additional instruments from user request (no local data, but include strategies)
ADDITIONAL_INSTRUMENTS = {
    "cryBTCUSD": {"name":"BTC/USD",  "type":"crypto", "strategies": VOLATILITY_STRATEGIES},  # Using vol strategies for crypto
    "frxXAUUSD": {"name":"XAU/USD",  "type":"metal",  "strategies": VOLATILITY_STRATEGIES},  # Using vol strategies for metals
    "CRASH1000": {"name":"Crash_1000", "type":"crash", "strategies": CRASH_STRATEGIES},
}

def main():
    all_strategies = {}
    total = 0

    for sym, info in {**INSTRUMENT_STRATEGY_MAP, **ADDITIONAL_INSTRUMENTS}.items():
        strats = []
        for s in info["strategies"]:
            strat = dict(s)
            strat["instrument"] = sym
            strat["instrument_name"] = info["name"]
            strat["instrument_type"] = info["type"]
            strat["full_id"] = f"{sym}_{s['id']}"
            strat["pip_size"] = 0.01
            strats.append(strat)

        all_strategies[sym] = strats
        total += len(strats)

        # Save per-instrument file
        with open(STRAT_DIR / f"{sym}_strategies.json", "w") as f:
            json.dump(strats, f, indent=2)

    # Save master index
    index = {sym: [s["full_id"] for s in strats] for sym, strats in all_strategies.items()}
    index["_total"] = total
    index["_instruments"] = list(all_strategies.keys())
    with open(STRAT_DIR / "strategy_index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Save all strategies in one file
    with open(STRAT_DIR / "all_strategies.json", "w") as f:
        json.dump(all_strategies, f, indent=2)

    print(f"Created {total} strategies across {len(all_strategies)} instruments")
    for sym, strats in all_strategies.items():
        print(f"  {sym}: {len(strats)} strategies")

if __name__ == "__main__":
    main()
