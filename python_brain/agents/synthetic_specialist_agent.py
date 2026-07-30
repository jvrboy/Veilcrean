"""
synthetic_specialist_agent.py
=============================
Agent 15 — Synthetic Specialist Agent

Focuses purely on Deriv's Synthetic indices: Volatilities, Jump, and 
specifically Drift Switch Indices (10, 20, 30).
"""
from __future__ import annotations
from typing import Any, Dict
from .base_agent import BaseAgent

class SyntheticSpecialistAgent(BaseAgent):
    skill_role = "analyst"
    name = "synthetic_specialist"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = context.get("snapshot")
        if not snapshot: return {}
        
        symbol = snapshot.symbol
        is_dsi = "DSI" in symbol or "Drift" in symbol
        
        if not is_dsi:
            return {"specialist_active": False}
            
        # DSI Specific Logic
        # These indices have specific math behind them.
        # Drift 10/20/30 refers to the 'drift' or bias intensity.
        
        dsi_tool = context.get("technical_report", {}).get("tool_results", {}).get("DriftSwitchTool")
        
        bias = 0.0
        if dsi_tool:
            # We look for "Clean Drift" (low volatility, consistent direction)
            if dsi_tool.features.get("dsi_volatility", 1.0) < 0.001:
                bias = dsi_tool.score * 1.2 # Boost score for clean drift
        
        return {
            "specialist_active": True,
            "dsi_intensity": symbol[-2:], # 10, 20, or 30
            "synthetic_bias": bias,
            "recommendation": "STRONG_FOLLOW" if abs(bias) > 0.7 else "WAIT"
        }
