"""
mean_reversion.py
=================
Mean Reversion Strategy: Trading back to the Volume POC or Moving Average.
"""
from __future__ import annotations
from typing import Any, Dict
from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def check_signal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tech = context.get("technical_report", {})
        results = tech.get("tool_results", {})
        
        vol_prof = results.get("VolumeProfileTool")
        mom      = results.get("MomentumVolumeTool")
        
        if not (vol_prof and mom):
            return {"action": "HOLD", "confidence": 0.0}

        # Logic: 
        # 1. Price is extended (RSI > 70 or < 30)
        # 2. Price is far from POC (Point of Control)
        # 3. Momentum is slowing down (MACD Histogram flattening)
        
        rsi = mom.features.get("mom_rsi_M15", 0.5) * 100 # Approx
        poc_dist = vol_prof.features.get("poc_dist", 0)
        
        if rsi > 80 and poc_dist > 0.0050: # Overbought + above POC
            return {"action": "SELL", "confidence": 0.7, "reason": "Mean Reversion to POC"}
        if rsi < 20 and poc_dist < -0.0050: # Oversold + below POC
            return {"action": "BUY", "confidence": 0.7, "reason": "Mean Reversion to POC"}
            
        return {"action": "HOLD", "confidence": 0.0}
