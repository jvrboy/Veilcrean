"""
trainer.py
==========
Training loops for the 3 networks.

Veilcrean retrains *offline* on accumulated trade-journal data using a
held-out validation split. New models are only deployed if they beat
the previous one on the holdout (see `Validator`).
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models.trade_decision_net  import TradeDecisionNet
from .models.risk_management_net import RiskManagementNet
from .models.regime_classifier   import RegimeClassifier
from ..config import NN_CFG


@dataclass
class TrainingReport:
    net_name:   str
    epochs:     int
    final_loss: float
    val_acc:    float
    val_loss:   float
    deployed:   bool = False


class Trainer:
    """Trains the 3 networks on a (X, y) dataset."""

    def __init__(self,
                 input_dim: int,
                 device:    str = "cpu",
                 lr:        float = None,
                 epochs:    int   = None,
                 batch:     int   = None):
        self.input_dim = input_dim
        self.device     = torch.device(device)
        self.lr         = lr     or NN_CFG.learning_rate
        self.epochs     = epochs or NN_CFG.epochs
        self.batch      = batch  or NN_CFG.batch_size

        # Instantiate networks
        self.trade_net  = TradeDecisionNet(input_dim).to(self.device)
        self.risk_net   = RiskManagementNet(input_dim).to(self.device)
        self.regime_net = RegimeClassifier(input_dim).to(self.device)

        # Optimizers
        self.opt_trade  = torch.optim.AdamW(self.trade_net.parameters(),  lr=self.lr, weight_decay=1e-4)
        self.opt_risk   = torch.optim.AdamW(self.risk_net.parameters(),   lr=self.lr, weight_decay=1e-4)
        self.opt_regime = torch.optim.AdamW(self.regime_net.parameters(), lr=self.lr, weight_decay=1e-4)

    # ------------------------------------------------------------------ API
    def train_trade_net(self, X: np.ndarray, y_action: np.ndarray, y_conf: np.ndarray) -> TrainingReport:
        """X: (N, F), y_action: (N,) with {0=BUY, 1=SELL, 2=HOLD}, y_conf: (N,) in [0,1]."""
        self.trade_net.train()
        Xt = torch.tensor(X,        dtype=torch.float32, device=self.device)
        yt = torch.tensor(y_action, dtype=torch.long,    device=self.device)
        ct = torch.tensor(y_conf,   dtype=torch.float32, device=self.device)

        ds = TensorDataset(Xt, yt, ct)
        loader = DataLoader(ds, batch_size=self.batch, shuffle=True, drop_last=False)

        loss_fn = nn.CrossEntropyLoss()
        last_loss = 0.0
        for ep in range(self.epochs):
            for xb, yb, cb in loader:
                logits, conf = self.trade_net(xb)
                loss = loss_fn(logits, yb) + 0.1 * F.mse_loss(conf, cb)
                self.opt_trade.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trade_net.parameters(), 1.0)
                self.opt_trade.step()
                last_loss = float(loss.item())
        return TrainingReport("trade_net", self.epochs, last_loss, 0.0, 0.0)

    def train_regime_net(self, X: np.ndarray, y_regime: np.ndarray) -> TrainingReport:
        self.regime_net.train()
        Xt = torch.tensor(X,        dtype=torch.float32, device=self.device)
        yt = torch.tensor(y_regime, dtype=torch.long,    device=self.device)
        ds = TensorDataset(Xt, yt)
        loader = DataLoader(ds, batch_size=self.batch, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        last_loss = 0.0
        for ep in range(self.epochs):
            for xb, yb in loader:
                logits = self.regime_net(xb)
                loss = loss_fn(logits, yb)
                self.opt_regime.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.regime_net.parameters(), 1.0)
                self.opt_regime.step()
                last_loss = float(loss.item())
        return TrainingReport("regime_net", self.epochs, last_loss, 0.0, 0.0)

    def train_risk_net(self, X: np.ndarray, actions: np.ndarray, confs: np.ndarray,
                       y_sl: np.ndarray, y_tp: np.ndarray, y_lot: np.ndarray) -> TrainingReport:
        self.risk_net.train()
        Xt = torch.tensor(X,        dtype=torch.float32, device=self.device)
        at = torch.tensor(actions,  dtype=torch.long,    device=self.device)
        ct = torch.tensor(confs,    dtype=torch.float32, device=self.device)
        sl = torch.tensor(y_sl,    dtype=torch.float32, device=self.device)
        tp = torch.tensor(y_tp,    dtype=torch.float32, device=self.device)
        lo = torch.tensor(y_lot,   dtype=torch.float32, device=self.device)

        ds = TensorDataset(Xt, at, ct, sl, tp, lo)
        loader = DataLoader(ds, batch_size=self.batch, shuffle=True)
        last_loss = 0.0
        for ep in range(self.epochs):
            for xb, ab, cb, slb, tpb, lob in loader:
                action_oh = F.one_hot(ab, num_classes=3).float()
                sl_pred, tp_pred, lo_pred = self.risk_net(xb, action_oh, cb)
                loss = F.mse_loss(sl_pred.squeeze(-1), slb) \
                     + F.mse_loss(tp_pred.squeeze(-1), tpb) \
                     + F.mse_loss(lo_pred.squeeze(-1), lob)
                self.opt_risk.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.risk_net.parameters(), 1.0)
                self.opt_risk.step()
                last_loss = float(loss.item())
        return TrainingReport("risk_net", self.epochs, last_loss, 0.0, 0.0)

    # ------------------------------------------------------------------ helpers
    def set_state(self, trade_state=None, risk_state=None, regime_state=None) -> None:
        if trade_state  is not None: self.trade_net.load_state_dict(trade_state)
        if risk_state   is not None: self.risk_net.load_state_dict(risk_state)
        if regime_state is not None: self.regime_net.load_state_dict(regime_state)

    def get_state(self) -> Dict[str, dict]:
        return {
            "trade":  copy.deepcopy(self.trade_net.state_dict()),
            "risk":   copy.deepcopy(self.risk_net.state_dict()),
            "regime": copy.deepcopy(self.regime_net.state_dict()),
        }

    def eval(self) -> None:
        self.trade_net.eval()
        self.risk_net.eval()
        self.regime_net.eval()
