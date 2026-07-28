"""
base_strategy.py
================
Base class for rule-based strategies.
"""
from __future__ import annotations
from typing import Any, Dict, Optional

class BaseStrategy:
    name = "base_strategy"

    def check_signal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a dict with:
            'action': 'BUY' | 'SELL' | 'HOLD'
            'confidence': float
            'reason': str
        """
        return {"action": "HOLD", "confidence": 0.0, "reason": "Not implemented"}
