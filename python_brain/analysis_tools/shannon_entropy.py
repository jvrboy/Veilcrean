"""
shannon_entropy.py
==================
Tool 155 — Shannon Entropy Regime Filter

Measures the information entropy of the recent return distribution. A
market near maximum entropy is effectively random — any directional
signal taken there is a coin flip, which is where most losses come
from. Low entropy means returns are structured (persistent or cyclic)
and directional signals are worth acting on.

Score  = direction of the drift, but scaled DOWN as entropy rises, so
         in random regimes this tool actively pulls the aggregate
         toward neutral (a built-in loss reducer).
Confidence = 1 - normalized entropy.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolResult


class ShannonEntropyTool(BaseTool):
    name = "shannon_entropy"

    N_BINS = 8
    LOOKBACK = 96

    def analyze(self, buffers: Dict[str, pd.DataFrame], **ctx) -> ToolResult:
        result = ToolResult(tool_name=self.name)
        df = buffers.get("M15") or buffers.get("H1")
        if df is None or len(df) < 40:
            return result

        closes = df["close"].to_numpy(dtype=float)[-(self.LOOKBACK + 1):]
        rets = np.diff(closes)
        sd = float(np.std(rets))
        if sd <= 0 or len(rets) < 20:
            return result

        # ---- binned entropy of standardized returns -------------------- #
        z = np.clip(rets / sd, -3, 3)
        hist, _ = np.histogram(z, bins=self.N_BINS, range=(-3, 3))
        p = hist / hist.sum()
        p = p[p > 0]
        entropy = float(-(p * np.log2(p)).sum())
        max_entropy = np.log2(self.N_BINS)
        norm_entropy = float(np.clip(entropy / max_entropy, 0, 1))

        # ---- sign entropy (persistence of direction) ------------------- #
        signs = (rets > 0).astype(int)
        p_up = signs.mean()
        p_up = min(max(p_up, 1e-9), 1 - 1e-9)
        sign_entropy = float(-(p_up * np.log2(p_up)
                               + (1 - p_up) * np.log2(1 - p_up)))

        structure = 1.0 - norm_entropy          # 0 = random, 1 = structured
        persistence = 1.0 - sign_entropy        # 0 = 50/50, 1 = one-sided

        drift = float(np.tanh(np.mean(rets) / (sd / np.sqrt(len(rets)) + 1e-12) * 0.5))

        # dampen the drift read by how random the market currently is
        score = drift * float(np.clip(0.3 + 0.7 * max(structure, persistence), 0, 1))
        confidence = float(np.clip(0.25 + 0.65 * max(structure, persistence), 0.1, 0.9))

        result.score = float(np.clip(score, -1, 1))
        result.confidence = confidence
        result.features = {
            "return_entropy_norm": norm_entropy,
            "sign_entropy": sign_entropy,
            "entropy_structure": structure,
            "entropy_drift": drift,
        }
        result.metadata = {
            "entropy_bits": entropy,
            "max_entropy_bits": float(max_entropy),
            "p_up": float(p_up),
            "regime": "RANDOM" if norm_entropy > 0.95 else
                      ("STRUCTURED" if norm_entropy < 0.80 else "MIXED"),
        }
        return result
