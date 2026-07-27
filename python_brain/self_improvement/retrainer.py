"""
retrainer.py
============
Periodic retraining. Reads closed trades from the journal, builds
training matrices, and retrains the three networks. Only deploys the
new model if it strictly improves on the holdout.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import SI_CFG, RISK_CFG
from ..neural_network.trainer  import Trainer
from ..neural_network.validator import Validator
from ..neural_network.model_manager import ModelManager, ModelBundle
from .trade_journal import TradeJournal


@dataclass
class RetrainResult:
    deployed: bool
    version:  str
    acc:      float
    message:  str


class Retrainer:
    """Coordinates the retraining cycle."""

    def __init__(self, journal: TradeJournal, model_manager: ModelManager, input_dim: int):
        self.journal = journal
        self.mm = model_manager
        self.input_dim = input_dim
        self.trainer: Optional[Trainer] = None
        self.validator = Validator()

    # ------------------------------------------------------------------ API
    def should_retrain(self) -> bool:
        n = self.journal.count_closed()
        if n < SI_CFG.min_trades_for_retrain:
            return False
        return self.journal.n_trades_since_last_train() >= SI_CFG.retrain_every_n_trades

    def run(self) -> RetrainResult:
        closed = self.journal.all_closed()
        if len(closed) < SI_CFG.min_trades_for_retrain:
            return RetrainResult(False, "", 0.0, "not enough data")

        # Build training matrices from journal
        X, y_action, y_conf, y_sl, y_tp, y_lot, y_regime = self._build_dataset(closed)
        if X.shape[0] < 50:
            return RetrainResult(False, "", 0.0, "feature matrix too small")

        # Train/val split
        n = X.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        split = int(n * 0.8)
        tr, va = idx[:split], idx[split:]
        if len(va) < 5:
            return RetrainResult(False, "", 0.0, "validation set too small")

        # 1. Init trainer
        self.trainer = Trainer(input_dim=self.input_dim)

        # 2. Load existing weights (warm start) if any
        current = self.mm.load_latest()
        if current:
            try:
                self.trainer.set_state(
                    trade_state=current.trade_state,
                    risk_state=current.risk_state,
                    regime_state=current.regime_state,
                )
            except Exception:
                pass

        # 3. Train
        self.trainer.train_trade_net(X[tr], y_action[tr], y_conf[tr])
        self.trainer.train_regime_net(X[tr], y_regime[tr])
        # one-hot actions for risk net
        import torch.nn.functional as F
        import torch
        actions_oh = F.one_hot(torch.tensor(y_action[tr]).long(), num_classes=3).numpy()
        self.trainer.train_risk_net(
            X[tr], y_action[tr], y_conf[tr],
            y_sl[tr], y_tp[tr], y_lot[tr]
        )

        # 4. Validate
        v_trade = self.validator.validate_trade(self.trainer.trade_net, X[va], y_action[va])
        v_regime= self.validator.validate_regime(self.trainer.regime_net, X[va], y_regime[va])
        avg_acc = (v_trade.accuracy + v_regime.accuracy) / 2.0

        # 5. Decide deploy
        deploy = v_trade.beats_holdout and v_regime.beats_holdout and avg_acc >= SI_CFG.min_performance_to_deploy
        version = ""
        if deploy:
            bundle = self.mm.save(
                self.trainer.trade_net, self.trainer.risk_net, self.trainer.regime_net,
                metadata={"acc": avg_acc, "n_samples": int(n),
                          "val_trade_acc": v_trade.accuracy,
                          "val_regime_acc": v_regime.accuracy}
            )
            version = bundle.version
            self.journal.log_retrain(version, int(n), avg_acc)
            msg = f"deployed {version} (acc={avg_acc:.3f})"
        else:
            msg = f"kept old model (acc={avg_acc:.3f} < threshold)"

        return RetrainResult(deploy, version, avg_acc, msg)

    # ------------------------------------------------------------------ internal
    @staticmethod
    def _build_dataset(records) -> Tuple[np.ndarray, ...]:
        """Translate TradeRecord list into NN training matrices."""
        X, y_act, y_conf, y_sl, y_tp, y_lot, y_reg = [], [], [], [], [], [], []
        REGIME_MAP = {"TRENDING": 0, "RANGING": 1, "VOLATILE": 2, "CHOPPY": 3, "BREAKOUT": 4}
        for r in records:
            if not r.feature_vec: continue
            X.append(r.feature_vec)
            y_act.append(0 if r.direction == "BUY" else 1 if r.direction == "SELL" else 2)
            y_conf.append(r.confidence)
            # SL/TP normalized by soft max pips
            sl_norm = min(max(abs(r.entry_price - r.sl) / 0.0100, 0), 1)  # 100 pips = 1.0
            tp_norm = min(max(abs(r.tp - r.entry_price) / 0.0300, 0), 1)  # 300 pips = 1.0
            lot_norm= min(max(r.lots / 1.0, 0), 1)
            y_sl.append(sl_norm)
            y_tp.append(tp_norm)
            y_lot.append(lot_norm)
            y_reg.append(REGIME_MAP.get(r.regime, 0))
        if not X:
            return (np.zeros((0, 1)),) * 7
        X = np.array(X, dtype=np.float32)
        return (X,
                np.array(y_act,  dtype=np.int64),
                np.array(y_conf, dtype=np.float32),
                np.array(y_sl,   dtype=np.float32),
                np.array(y_tp,   dtype=np.float32),
                np.array(y_lot,  dtype=np.float32),
                np.array(y_reg,  dtype=np.int64))
