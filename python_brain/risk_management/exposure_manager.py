"""
exposure_manager.py
====================
Enforces portfolio-level limits: max open positions, correlation, etc.
"""
from __future__ import annotations
from typing import List

from ..config import RISK_CFG


class ExposureManager:
    def __init__(self):
        self.positions: List[dict] = []

    # ------------------------------------------------------------------ API
    def sync(self, positions: List[dict]) -> None:
        """Sync internal view with the EA's current positions."""
        self.positions = list(positions or [])

    def can_open(self, symbol: str, direction: str) -> bool:
        if len(self.positions) >= RISK_CFG.max_open_positions:
            return False
        # correlation rule: no 2 same-direction positions on correlated pairs
        same_dir = [p for p in self.positions
                    if p.get("type") == direction and self._correlated(p.get("symbol", ""), symbol)]
        if len(same_dir) >= RISK_CFG.max_correlated_positions:
            return False
        return True

    def open_positions_count(self) -> int:
        return len(self.positions)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _correlated(a: str, b: str) -> bool:
        """Simple group lookup — pairs in the same group are correlated."""
        groups = {
            "EUR": ["EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD"],
            "GBP": ["GBPUSD", "EURGBP", "GBPJPY", "GBPCHF", "GBPAUD"],
            "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY"],
            "AUD": ["AUDUSD", "EURAUD", "GBPAUD", "AUDJPY", "AUDNZD", "AUDCAD"],
            "CAD": ["USDCAD", "EURCAD", "GBPCAD", "CADJPY", "AUDCAD"],
            "CHF": ["USDCHF", "EURCHF", "GBPCHF", "CHFJPY"],
            "USD": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"],
        }
        for syms in groups.values():
            if a in syms and b in syms:
                return True
        return False
