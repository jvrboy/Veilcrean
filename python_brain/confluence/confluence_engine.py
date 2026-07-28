"""
confluence_engine.py
====================
Runs all 8 analysis tools on the current buffer state, then hands
their results to the FeatureBuilder.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np

from ..analysis_tools import ALL_TOOLS, ToolResult
from ..analysis_tools.base_tool import BaseTool
from ..communication.data_parser import MarketSnapshot
from ..communication.data_parser import mid_price
from .feature_builder import FeatureBuilder


class ConfluenceEngine:
    """Owns the 8 tools and produces feature vectors."""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {t.__name__: t() for t in ALL_TOOLS}
        self.builder = FeatureBuilder()

    # ------------------------------------------------------------------ API
    def run(self, snapshot: MarketSnapshot, buffers, extra_ctx: Optional[dict] = None) -> Dict:
        """Run all tools, build feature vector, return everything."""
        price = mid_price(snapshot)
        pip_size = self._infer_pip_size(price)

        now = snapshot.timestamp or datetime.now(timezone.utc)
        ctx = {
            "price":    price,
            "pip_size": pip_size,
            "now":      now,
            "hour":     now.hour,
            "weekday":  now.weekday(),
            "open_positions": len(snapshot.positions),
            "spread":   snapshot.tick.spread if snapshot.tick else 0.0,
        }
        if extra_ctx:
            ctx.update(extra_ctx)

        results: Dict[str, ToolResult] = {}
        for name, tool in self.tools.items():
            if name == "AIReasonerTool": continue # Run AI last
            try:
                res = tool.analyze(buffers, **ctx)
            except Exception as e:
                res = ToolResult(tool_name=name)
                res.errors.append(f"exception: {e}")
            results[name] = res

        # Run AI Reasoner with context of other tools
        if "AIReasonerTool" in self.tools:
            ctx["prev_scores"] = {n: r.score for n, r in results.items()}
            ctx["symbol"] = snapshot.symbol
            try:
                results["AIReasonerTool"] = self.tools["AIReasonerTool"].analyze(buffers, **ctx)
            except Exception as e:
                results["AIReasonerTool"] = ToolResult(tool_name="AIReasonerTool", errors=[str(e)])

        feature_vec = self.builder.build(results, ctx)

        # Aggregate score = simple weighted average
        agg = 0.0
        total_w = 0.0
        for name, res in results.items():
            w = max(0.1, res.confidence)
            agg += res.score * w
            total_w += w
        agg_score = agg / max(total_w, 1e-9)

        return {
            "tool_results":  results,
            "feature_vector": feature_vec,
            "feature_names":  self.builder.names(),
            "aggregate_score": float(np.clip(agg_score, -1, 1)),
            "context":        ctx,
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _infer_pip_size(price: float) -> float:
        if price <= 0: return 0.0001
        if "JPY" in str(price): return 0.01
        if price > 1000:        return 0.1     # indices / metals
        return 0.0001
