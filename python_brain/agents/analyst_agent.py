"""
analyst_agent.py
================
Orchestrates the 10+ technical analysis tools and produces
a consolidated technical report.
"""
from __future__ import annotations
from typing import Any, Dict, List

from .base_agent import BaseAgent
from ..confluence.confluence_engine import ConfluenceEngine

class AnalystAgent(BaseAgent):
    name = "analyst_agent"

    def __init__(self):
        super().__init__()
        self.confluence = ConfluenceEngine()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = context.get("snapshot")
        buffers = context.get("buffers")
        
        if snapshot is None or buffers is None:
            return {"error": "Missing data for analysis"}

        self.log.debug(f"Analyzing {snapshot.symbol}")
        analysis_results = self.confluence.run(snapshot, buffers)
        
        return {
            "technical_report": analysis_results,
            "aggregate_score": analysis_results.get("aggregate_score", 0.0),
            "features": analysis_results.get("feature_vector")
        }
