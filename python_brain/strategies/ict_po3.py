"""
ict_po3.py
==========
ICT Power of 3 (PO3) Strategy: Accumulation, Manipulation, Distribution.
Focuses on session opens and liquidity sweeps.
"""
from __future__ import annotations
from typing import Any, Dict
from .base_strategy import BaseStrategy

class ICTPO3Strategy(BaseStrategy):
    name = "ict_po3"

    def check_signal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tech = context.get("technical_report", {})
        results = tech.get("tool_results", {})
        
        # We look for Session + Liquidity + Market Structure
        sess = results.get("SessionTimeTool")
        liq  = results.get("LiquidityTool")
        ms   = results.get("MarketStructureTool")
        
        if not (sess and liq and ms):
            return {"action": "HOLD", "confidence": 0.0}

        # PO3 Logic: 
        # 1. In Killzone? (Accumulation/Manipulation phase)
        # 2. Sweep of Liquidity? (Manipulation)
        # 3. Market Structure Shift (BOS) in opposite direction? (Distribution)
        
        is_killzone = sess.metadata.get("is_killzone", False)
        # Assuming LiquidityTool features contains sweep info
        sweep_high = liq.features.get("liq_sweep_high", 0) > 0
        sweep_low  = liq.features.get("liq_sweep_low", 0) > 0
        
        if is_killzone:
            if sweep_low and ms.score > 0.5: # Sweep low then bullish BOS
                return {"action": "BUY", "confidence": 0.85, "reason": "PO3 Bullish Reversal"}
            if sweep_high and ms.score < -0.5: # Sweep high then bearish BOS
                return {"action": "SELL", "confidence": 0.85, "reason": "PO3 Bearish Reversal"}
                
        return {"action": "HOLD", "confidence": 0.0}
