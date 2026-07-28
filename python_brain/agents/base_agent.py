"""
base_agent.py
=============
Base class for all Veilcrean Agents.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
from ..utils.logger import get_logger

class BaseAgent:
    name = "base_agent"

    def __init__(self):
        self.log = get_logger(self.name)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the context and return insights/decisions."""
        raise NotImplementedError
