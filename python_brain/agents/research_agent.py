"""
research_agent.py
=================
Agent that performs long-term analysis and historical pattern matching.
Finds "Fractal Similarity" to previous market setups.
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from .base_agent import BaseAgent

class ResearchAgent(BaseAgent):
    name = "research_agent"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        buffers = context.get("buffers")
        if not buffers: return {}
        
        # Simple fractal search:
        # Compare current last 10 candles with historical data
        current_tf = "H1"
        df = buffers.get(current_tf)
        if df is None or len(df) < 200: return {}
        
        current_pattern = df["close"].tail(10).values
        # Normalize pattern
        current_pattern = (current_pattern - np.mean(current_pattern)) / (np.std(current_pattern) + 1e-9)
        
        # Search back in time
        best_match_corr = -1.0
        match_outcome = 0.0
        
        # This is a heavy operation, usually we'd only do this occasionally
        # but for this demo, we check a small window
        history = df["close"].values[:-10]
        for i in range(len(history) - 10):
            window = history[i:i+10]
            window_norm = (window - np.mean(window)) / (np.std(window) + 1e-9)
            corr = np.corrcoef(current_pattern, window_norm)[0, 1]
            if corr > best_match_corr:
                best_match_corr = corr
                # What happened after this match? (next 5 candles)
                if i + 15 < len(history):
                    after = history[i+10:i+15]
                    match_outcome = (after[-1] - after[0]) / after[0]

        return {
            "fractal_similarity": best_match_corr,
            "historical_outcome": float(match_outcome),
            "research_bias": 1.0 if match_outcome > 0 and best_match_corr > 0.8 else -1.0 if match_outcome < 0 and best_match_corr > 0.8 else 0.0
        }
