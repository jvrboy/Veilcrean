"""
main.py
=======
Veilcrean Python Brain — main loop.

Run with:

    python -m python_brain.main

The brain connects to the MT5 EA via ZMQ, runs the 8 analysis tools,
asks the 3 neural networks for a decision, and (if confidence is
above the dynamic threshold and all safety checks pass) sends a
trade command back to the EA.

The loop is single-threaded and event-driven by ZMQ polling. On every
new packet from the EA we:

    1. parse it into a MarketSnapshot
    2. push its candles into the rolling buffer
    3. run the confluence engine
    4. ask Network A for an action + confidence
    5. ask Network C for the regime
    6. ask Network B for SL/TP/lot
    7. apply risk management
    8. if all checks pass, send a TRADE_COMMAND to the EA
    9. log the trade to the journal
   10. periodically retrain the networks
"""
from __future__ import annotations
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    ZMQ_CFG, NN_CFG, SI_CFG, RISK_CFG, ALERT_CFG, DERIV_CFG,
    TIMEFRAMES, CANDLE_HISTORY, JOURNAL_DB, MODELS_DIR,
)
from .communication            import ZMQServer, DataParser
from .communication.deriv_client import DerivClient
from .preprocessor              import BufferManager
from .confluence                import ConfluenceEngine
from .neural_network            import (
    TradeDecisionNet, RiskManagementNet, RegimeClassifier,
    ModelManager, Trainer, Validator,
)
from .neural_network.models.regime_classifier import REGIME_LABELS
from .self_improvement          import (
    TradeJournal, TradeRecord, PerformanceTracker, Retrainer, ThresholdAdjuster,
)
from .risk_management           import PositionSizer, DrawdownGuard, ExposureManager, TrailingManager
from .agents.coordinator_agent  import CoordinatorAgent
from .utils                     import get_logger, Alerter, Visualizer
from .communication.data_parser import mid_price

log = get_logger("veilcrean")
alerter = Alerter()


# ============================================================================
# DecisionEngine
# ============================================================================
class DecisionEngine:
    """Combines the 3 networks into a single (action, sl, tp, lot) decision."""

    def __init__(self, input_dim: int, model_manager: ModelManager, device: str = "cpu"):
        self.input_dim = input_dim
        self.mm = model_manager
        self.device = torch.device(device)
        self.trade  = TradeDecisionNet(input_dim).to(self.device)
        self.risk   = RiskManagementNet(input_dim).to(self.device)
        self.regime = RegimeClassifier(input_dim).to(self.device)
        self._load_latest()

    def _load_latest(self) -> None:
        bundle = self.mm.load_latest()
        if bundle is None:
            log.warning("no model bundle found — using fresh (untrained) networks")
            return
        try:
            self.trade.load_state_dict(bundle.trade_state)
            self.risk.load_state_dict(bundle.risk_state)
            self.regime.load_state_dict(bundle.regime_state)
            log.info(f"loaded model bundle {bundle.version}")
        except Exception as e:
            log.error(f"failed to load bundle {bundle.version}: {e}")

    @torch.no_grad()
    def decide(self, feature_vec: np.ndarray) -> dict:
        """Return a decision dict with action, confidence, sl, tp, lot, regime."""
        # eval mode (BatchNorm requires >1 sample in training mode)
        self.trade.eval(); self.risk.eval(); self.regime.eval()
        x = torch.tensor(feature_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, conf = self.trade(x)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        action_idx = int(np.argmax(probs))
        action = ["BUY", "SELL", "HOLD"][action_idx]
        confidence = float(conf.item())
        action_oh = F.one_hot(torch.tensor([action_idx]), num_classes=3).float()
        sl_norm, tp_norm, lot_norm = self.risk(x, action_oh, torch.tensor([confidence]))
        regime_logits = self.regime(x)
        regime_idx = int(np.argmax(regime_logits, dim=-1)[0].item())
        regime = REGIME_LABELS[regime_idx] if regime_idx < len(REGIME_LABELS) else "UNKNOWN"
        return {
            "action":      action,
            "action_idx":  action_idx,
            "probs":       probs.tolist(),
            "confidence":  confidence,
            "sl_norm":     float(sl_norm.item()),
            "tp_norm":     float(tp_norm.item()),
            "lot_norm":    float(lot_norm.item()),
            "regime":      regime,
        }


# ============================================================================
# Veilcrean brain
# ============================================================================
class VeilcreanBrain:
    """The orchestrator."""

    def __init__(self):
        log.info("=" * 70)
        log.info("  Veilcrean — Python Brain starting up")
        log.info("=" * 70)

        # communication
        self.zmq = ZMQServer()
        self.zmq.start()
        self.deriv = None
        if DERIV_CFG.enabled:
            self.deriv = DerivClient(DERIV_CFG.app_id, DERIV_CFG.api_token)
            
        self.parser = DataParser()

        # preprocessing
        self.buffer = BufferManager(max_len=CANDLE_HISTORY)

        # confluence + decision
        self.confluence = ConfluenceEngine()
        # input_dim is set after first run
        self.input_dim = 64
        self.last_feature_vec: Optional[np.ndarray] = None
        self.mm = ModelManager()
        self.decision = DecisionEngine(self.input_dim, self.mm)
        
        # New Agent-based Coordinator
        self.coordinator = CoordinatorAgent(self.decision)

        # journal / performance
        self.journal = TradeJournal()
        self.perf    = PerformanceTracker(self.journal)
        self.retrainer = Retrainer(self.journal, self.mm, self.input_dim)
        self.threshold = ThresholdAdjuster()

        # risk
        self.sizer    = PositionSizer()
        self.guard    = DrawdownGuard()
        self.exposure = ExposureManager()
        self.trailing = TrailingManager()

        # misc
        self.vis = Visualizer()
        self.running = True
        self.last_feature_names: list = []
        self.last_decision: dict = {}
        self._register_signals()

    # ------------------------------------------------------------------ lifecycle
    def _register_signals(self):
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, *_):
        log.warning("shutdown signal received")
        self.running = False
        self.zmq.stop()
        sys.exit(0)

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        log.info("entering main loop")
        cycle = 0
        while self.running:
            try:
                raw = self.zmq.receive_market_data()
                if raw is None:
                    self._heartbeat_check()
                    self._maybe_retrain()
                    time.sleep(0.05)
                    continue

                self._handle_packet(raw)
                cycle += 1
                if cycle % 200 == 0:
                    self._publish_status()
            except Exception as e:
                log.exception(f"loop error: {e}")
                alerter.error(f"loop error: {e}")
                time.sleep(0.5)

    # ------------------------------------------------------------------ per-packet
    def _handle_packet(self, raw: dict) -> None:
        snapshot = self.parser.parse(raw)
        if snapshot is None:
            return
        
        # 1. Update buffers
        self.buffer.update(snapshot.candles)
        
        # 2. Run Agent Orchestration
        agent_ctx = {
            "snapshot": snapshot,
            "buffers":  self.buffer.all(),
        }
        report = self.coordinator.run(agent_ctx)
        
        decision = report["decision"]
        result   = report["technical_report"]
        self.last_feature_vec = result["feature_vector"]
        self.last_feature_names = result["feature_names"]

        # Lazy-init input_dim
        if self.last_feature_vec.shape[0] != self.input_dim:
            self.input_dim = int(self.last_feature_vec.shape[0])
            self.decision = DecisionEngine(self.input_dim, self.mm)
            self.retrainer = Retrainer(self.journal, self.mm, self.input_dim)
            self.coordinator = CoordinatorAgent(self.decision) # Re-init with new engine

        self.last_decision = decision
        ctx = result["context"]
        self._publish_status(result, decision)

        # 3. Dynamic Management (Trailing)
        self._manage_active_positions(snapshot)

        # 4. Risk & Threshold Checks
        if not report["risk_ok"]:
            if "KILL SWITCH" in report["risk_reason"]:
                self.zmq.send_trade_command({"action": "FLATTEN_ALL", "symbol": snapshot.symbol})
            return

        threshold = self.threshold.update()
        if decision["action"] == "HOLD":
            return
        if decision["confidence"] < threshold:
            log.debug(f"below threshold ({decision['confidence']:.2f} < {threshold:.2f})")
            return

        # 5. Build & send command
        cmd = self._build_trade_command(snapshot, decision, ctx)
        if cmd is None:
            return
        
        log.info(f"📤 AGENT-APPROVED TRADE: {cmd}")
        sent = self.zmq.send_trade_command(cmd)
        if sent:
            self._journal_open(snapshot, decision, ctx, cmd)

    def _manage_active_positions(self, snapshot) -> None:
        """Dynamic management of open trades (trailing stops, etc.)"""
        if not snapshot.positions:
            return

        for pos in snapshot.positions:
            # We only manage positions for the current symbol
            if pos.symbol != snapshot.symbol:
                continue

            current_price = (snapshot.tick.bid + snapshot.tick.ask) / 2.0
            new_sl = self.trailing.compute_trailing_stop(
                pos.symbol, pos.type, current_price, pos.price_open, self.buffer.all()
            )

            if new_sl:
                # Basic check: only move SL in our favor
                if pos.type == "BUY" and new_sl > pos.sl + 0.0001:
                    log.info(f"Trailing SL for {pos.symbol} to {new_sl:.5f}")
                    self.zmq.send_trade_command({
                        "action": "MODIFY",
                        "ticket": pos.ticket,
                        "sl": round(new_sl, 5),
                        "tp": pos.tp
                    })
                elif pos.type == "SELL" and new_sl < pos.sl - 0.0001:
                    log.info(f"Trailing SL for {pos.symbol} to {new_sl:.5f}")
                    self.zmq.send_trade_command({
                        "action": "MODIFY",
                        "ticket": pos.ticket,
                        "sl": round(new_sl, 5),
                        "tp": pos.tp
                    })

    # ------------------------------------------------------------------ helpers
    def _risk_ok(self, snapshot, decision) -> bool:
        # 1. EA heartbeat
        if not self.zmq.heartbeat_is_alive(RISK_CFG.heartbeat_timeout_sec):
            alerter.kill_switch("EA heartbeat lost")
            return False
        # 2. drawdown guard
        if snapshot.account:
            self.guard.update(snapshot.account.equity)
            if self.guard.kill_switch:
                alerter.kill_switch(self.guard.kill_reason)
                # flatten
                self.zmq.send_trade_command({"action": "FLATTEN_ALL", "symbol": snapshot.symbol})
                return False
        # 3. spread filter
        if snapshot.tick and snapshot.tick.spread > RISK_CFG.max_spread_points:
            log.debug(f"spread too wide: {snapshot.tick.spread}")
            return False
        # 4. exposure
        self.exposure.sync([
            {"symbol": p.symbol, "type": p.type}
            for p in snapshot.positions
        ])
        if not self.exposure.can_open(snapshot.symbol, decision["action"]):
            log.debug("exposure limits block new trade")
            return False
        return True

    def _build_trade_command(self, snapshot, decision, ctx) -> Optional[dict]:
        if snapshot.account is None or snapshot.tick is None:
            return None
        # SL / TP in pips (soft limits applied)
        sl_pips = max(RISK_CFG.sl_min_pips,
                      min(RISK_CFG.sl_max_pips, decision["sl_norm"] * RISK_CFG.sl_max_pips))
        tp_pips = max(RISK_CFG.tp_min_pips,
                      min(RISK_CFG.tp_max_pips, decision["tp_norm"] * RISK_CFG.tp_max_pips))
        pip = 0.0001 if (snapshot.tick.bid + snapshot.tick.ask) / 2 > 10 else 0.01
        price = (snapshot.tick.bid + snapshot.tick.ask) / 2.0
        if decision["action"] == "BUY":
            sl = price - sl_pips * pip
            tp = price + tp_pips * pip
        else:
            sl = price + sl_pips * pip
            tp = price - tp_pips * pip
        lots = self.sizer.lots(
            account_balance=snapshot.account.balance,
            sl_pips=sl_pips,
            risk_pct=RISK_CFG.max_risk_per_trade_pct,
        )
        # confidence-modulated lot multiplier
        lots *= max(0.25, decision["lot_norm"])  # never zero
        lots = float(np.clip(lots, RISK_CFG.lot_min, RISK_CFG.lot_max))
        return {
            "type":       "TRADE_COMMAND",
            "action":     "OPEN",
            "direction":  decision["action"],
            "symbol":     snapshot.symbol,
            "lot_size":   round(lots, 2),
            "sl":         round(sl, 5),
            "tp":         round(tp, 5),
            "confidence": round(decision["confidence"], 3),
            "strategy_tag": f"veilcrean_v1",
        }

    def _journal_open(self, snapshot, decision, ctx, cmd) -> None:
        if snapshot.tick is None or snapshot.account is None: return
        price = (snapshot.tick.bid + snapshot.tick.ask) / 2.0
        rec = TradeRecord(
            trade_id    = str(uuid.uuid4())[:12],
            symbol      = snapshot.symbol,
            direction   = decision["action"],
            opened_at   = time.time(),
            entry_price = price,
            sl          = cmd["sl"],
            tp          = cmd["tp"],
            lots        = cmd["lot_size"],
            confidence  = decision["confidence"],
            regime      = decision["regime"],
            session     = ctx.get("hour", 0) and ("london" if 8 <= ctx["hour"] < 16 else "ny" if 13 <= ctx["hour"] < 22 else "asian"),
            weekday     = ctx.get("weekday", 0),
            strategy_tag= cmd.get("strategy_tag", ""),
            feature_vec = self.decision_input_to_list(),
        )
        self.journal.open_trade(rec)
        alerter.trade_open(snapshot.symbol, decision["action"], cmd["lot_size"],
                           cmd["sl"], cmd["tp"], decision["confidence"])

    def decision_input_to_list(self) -> list:
        """Helper to fetch last feature vector as plain list (for journal)."""
        if self.last_feature_vec is None:
            return []
        return self.last_feature_vec.tolist()

    # ------------------------------------------------------------------ maintenance
    def _heartbeat_check(self) -> None:
        # if no packet for a while but we haven't decided, do nothing
        pass

    def _maybe_retrain(self) -> None:
        if not self.retrainer.should_retrain():
            return
        log.info("starting retrain cycle…")
        try:
            res = self.retrainer.run()
            log.info(f"retrain result: {res}")
            if res.deployed:
                alerter.retrain(res.version, res.acc)
                # Reload the newly-deployed model
                self.decision = DecisionEngine(self.input_dim, self.mm)
        except Exception as e:
            log.exception(f"retrain failed: {e}")
            alerter.error(f"retrain failed: {e}")

    def _publish_status(self, result: Optional[dict] = None, decision: Optional[dict] = None) -> None:
        snap = self.perf.snapshot()
        status = {
            "type":         "BRAIN_STATUS",
            "timestamp":    time.time(),
            "win_rate":     snap.win_rate,
            "profit_factor":snap.profit_factor,
            "sharpe":       snap.sharpe,
            "max_dd_pct":   snap.max_drawdown_pct,
            "trades":       snap.total_trades,
            "threshold":    self.threshold.current,
            "kill_switch":  self.guard.kill_switch,
            "regime":       decision.get("regime", "—") if decision else "—",
            "confidence":   decision.get("confidence", 0.0) if decision else 0.0,
        }
        self.zmq.publish_status(status)


# ============================================================================
# Entry point
# ============================================================================
def main() -> int:
    brain = VeilcreanBrain()
    brain.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
