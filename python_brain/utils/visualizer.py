"""
visualizer.py
=============
Quick-and-dirty dashboard helpers (matplotlib-based). The full Dash
dashboard lives in a separate module under `dashboard/` if you want
a live web UI.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False

from ..config import BACKTESTS


class Visualizer:
    """Generates static charts (equity curve, drawdown, win-rate, etc.)."""

    def __init__(self, out_dir: Path = BACKTESTS):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public
    def equity_curve(self, pnls: List[float], name: str = "equity.png") -> Optional[Path]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        eq = np.cumsum(pnls)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eq, color="#4ec9b0", linewidth=1.6)
        ax.fill_between(range(len(eq)), eq, alpha=0.15, color="#4ec9b0")
        ax.set_title("Veilcrean — Equity Curve")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(True, alpha=0.3)
        out = self.out_dir / name
        fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
        return out

    def drawdown(self, pnls: List[float], name: str = "drawdown.png") -> Optional[Path]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        eq  = np.cumsum(pnls)
        peak= np.maximum.accumulate(eq)
        dd  = (eq - peak) / np.maximum(peak, 1e-9) * 100
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(range(len(dd)), dd, color="#ff6b6b", alpha=0.6)
        ax.set_title("Drawdown %")
        ax.grid(True, alpha=0.3)
        out = self.out_dir / name
        fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
        return out

    def feature_importance(self, names: List[str], importances: np.ndarray,
                           name: str = "feature_importance.png") -> Optional[Path]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        idx = np.argsort(np.abs(importances))[-20:]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(idx)), importances[idx], color="#4ec9b0")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([names[i] for i in idx])
        ax.set_title("Top 20 Feature Importances")
        out = self.out_dir / name
        fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
        return out
