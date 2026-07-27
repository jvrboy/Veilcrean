"""
base_tool.py
============
Abstract base class for analysis tools. Every tool produces a
``ToolResult`` with:

    score       — in [-1, 1];  +1 = strong buy, -1 = strong sell, 0 = neutral
    confidence  — in [0, 1];  how sure are we about the score
    metadata    — dict with diagnostic info (logged but not fed to NN)
    features    — dict of numeric features (fed directly to NN as inputs)

The NN doesn't see the human-readable metadata — it sees a flat
numeric feature vector assembled by the confluence builder.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ToolResult:
    tool_name: str
    score:     float = 0.0
    confidence: float = 0.5
    features:  Dict[str, float] = field(default_factory=dict)
    metadata:  Dict[str, Any]   = field(default_factory=dict)
    errors:    List[str]        = field(default_factory=list)

    def is_valid(self) -> bool:
        return not self.errors

    def to_feature_vector(self) -> List[float]:
        return list(self.features.values())


class BaseTool:
    """Subclass this to build a new analysis tool."""

    name: str = "base"

    def __init__(self, **kwargs):
        self.config = kwargs

    # Subclasses implement this
    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        raise NotImplementedError

    # Default helper: take the latest close of a TF (or NaN)
    @staticmethod
    def last_close(df: Optional[pd.DataFrame]) -> float:
        if df is None or df.empty or "close" not in df.columns: return float("nan")
        return float(df["close"].iloc[-1])

    @staticmethod
    def safe_min(d: Dict[str, float]) -> float:
        v = min(d.values()) if d else float("inf")
        return v if v != float("inf") else float("nan")
