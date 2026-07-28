"""
perpetual_scalper.py
====================
Strategy: Perpetual High-Frequency Scalper

Designed for 24/7 trading on Synthetics, Crypto, and Forex.
Always looks for micro-movements and tight exits.
"""
from __future__ import annotations
from typing import Any, Dict
from .base_strategy import BaseStrategy

class PerpetualScalperStrategy(BaseStrategy):
    name = "perpetual_scalper"

    def check_signal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tech = context.get("technical_report", {})
        results = tech.get("tool_results", {})
        
        # Core Scalping Tools
        micro = results.get("MicroTrendTool")
        bands = results.get("VolatilityBandsTool")
        smt   = results.get("SMTDivergenceTool")
        
        if not (micro and bands):
            return {"action": "HOLD", "confidence": 0.0}

        # Logic for perpetual entries
        # 1. Micro-trend confirms momentum
        # 2. Price is bouncing from volatility bands
        # 3. Aggregated score is strong
        
        agg_score = tech.get("aggregate_score", 0)
        
        # Scalping Thresholds (Very aggressive)
        if agg_score > 0.3 and micro.score > 0.5:
             return {"action": "BUY", "confidence": 0.8, "reason": "Scalp Trend Long"}
        
        if agg_score < -0.3 and micro.score < -0.5:
             return {"action": "SELL", "confidence": 0.8, "reason": "Scalp Trend Short"}
             
        # Mean Reversion Scalp
        if bands.score > 0.8: # Extremely Oversold
             return {"action": "BUY", "confidence": 0.7, "reason": "Scalp Mean Rev Long"}
        
        if bands.score < -0.8: # Extremely Overbought
             return {"action": "SELL", "confidence": 0.7, "reason": "Scalp Mean Rev Short"}

        return {"action": "HOLD", "confidence": 0.0}
