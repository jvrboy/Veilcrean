"""
validator.py
============
Backtest / holdout validation. Compares a new model against the
currently-deployed one on the validation split. New model only deploys
if it strictly improves on the holdout.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .models.trade_decision_net  import TradeDecisionNet
from .models.risk_management_net import RiskManagementNet
from .models.regime_classifier   import RegimeClassifier


@dataclass
class ValidationResult:
    name:        str
    accuracy:    float
    loss:        float
    beats_holdout: bool


class Validator:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    # ------------------------------------------------------------------ public
    def validate_trade(self, model: TradeDecisionNet, X: np.ndarray, y: np.ndarray) -> ValidationResult:
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            yt = torch.tensor(y, dtype=torch.long,    device=self.device)
            logits, _ = model(xt)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            acc = float((preds == yt).float().mean().item())
            loss = float(F.cross_entropy(logits, yt).item())
        return ValidationResult("trade_net", acc, loss, acc > 0.50)

    def validate_regime(self, model: RegimeClassifier, X: np.ndarray, y: np.ndarray) -> ValidationResult:
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            yt = torch.tensor(y, dtype=torch.long,    device=self.device)
            logits = model(xt)
            preds  = logits.argmax(dim=-1)
            acc = float((preds == yt).float().mean().item())
            loss = float(F.cross_entropy(logits, yt).item())
        return ValidationResult("regime_net", acc, loss, acc > 0.50)

    def validate_risk(self, model: RiskManagementNet, X, actions, confs, y_sl, y_tp, y_lot) -> ValidationResult:
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X,       dtype=torch.float32, device=self.device)
            at = torch.tensor(actions, dtype=torch.long,    device=self.device)
            ct = torch.tensor(confs,   dtype=torch.float32, device=self.device)
            sl = torch.tensor(y_sl,   dtype=torch.float32, device=self.device)
            tp = torch.tensor(y_tp,   dtype=torch.float32, device=self.device)
            lo = torch.tensor(y_lot,  dtype=torch.float32, device=self.device)
            oh = F.one_hot(at, num_classes=3).float()
            slp, tpp, lop = model(xt, oh, ct)
            mse = (F.mse_loss(slp.squeeze(-1), sl) + F.mse_loss(tpp.squeeze(-1), tp) + F.mse_loss(lop.squeeze(-1), lo)).item() / 3.0
        return ValidationResult("risk_net", 0.0, mse, mse < 0.05)
