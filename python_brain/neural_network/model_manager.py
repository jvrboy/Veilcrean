"""
model_manager.py
================
Save / load / version the three networks. Every deploy creates a new
file so we can always roll back.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from ..config import MODELS_DIR, SI_CFG
from .models.trade_decision_net  import TradeDecisionNet
from .models.risk_management_net import RiskManagementNet
from .models.regime_classifier   import RegimeClassifier


@dataclass
class ModelBundle:
    version:    str
    path:       str
    trade_state:  dict
    risk_state:   dict
    regime_state: dict
    metadata:   Dict


class ModelManager:
    """Owns on-disk model files and the currently-deployed bundle."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.dir = Path(models_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._current: Optional[ModelBundle] = None

    # ------------------------------------------------------------------ API
    def save(self, trade: TradeDecisionNet, risk: RiskManagementNet, regime: RegimeClassifier,
             metadata: Optional[Dict] = None) -> ModelBundle:
        ts = time.strftime("%Y%m%d_%H%M%S")
        version = f"{SI_CFG.model_version_prefix}{ts}"
        path = self.dir / f"bundle_{version}.pt"
        bundle = ModelBundle(
            version=version,
            path=str(path),
            trade_state=trade.state_dict(),
            risk_state=risk.state_dict(),
            regime_state=regime.state_dict(),
            metadata=metadata or {},
        )
        torch.save({
            "version":    version,
            "trade":      trade.state_dict(),
            "risk":       risk.state_dict(),
            "regime":     regime.state_dict(),
            "metadata":   metadata or {},
        }, path)
        self._current = bundle
        return bundle

    def load_latest(self) -> Optional[ModelBundle]:
        files = sorted(self.dir.glob("bundle_*.pt"))
        if not files: return None
        return self._load_path(files[-1])

    def load_version(self, version: str) -> Optional[ModelBundle]:
        files = list(self.dir.glob(f"bundle_{version}*.pt"))
        if not files: return None
        return self._load_path(files[0])

    def _load_path(self, path: Path) -> ModelBundle:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        bundle = ModelBundle(
            version=ckpt["version"],
            path=str(path),
            trade_state=ckpt["trade"],
            risk_state=ckpt["risk"],
            regime_state=ckpt["regime"],
            metadata=ckpt.get("metadata", {}),
        )
        self._current = bundle
        return bundle

    @property
    def current(self) -> Optional[ModelBundle]:
        return self._current

    def list_versions(self):
        return sorted(p.name for p in self.dir.glob("bundle_*.pt"))
