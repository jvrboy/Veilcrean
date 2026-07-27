"""
position_sizer.py
=================
Compute lot size from account balance, SL distance, and risk-per-trade %.
"""
from __future__ import annotations
from typing import Optional

from ..config import RISK_CFG


class PositionSizer:
    """Position sizing utility."""

    @staticmethod
    def lots(account_balance: float, sl_pips: float,
             risk_pct: Optional[float] = None,
             pip_value_per_lot: float = 10.0,
             contract_size:    float = 100000.0) -> float:
        """Return a lot size respecting the hard risk cap.

        pip_value_per_lot  — $ value of a 1-pip move for 1 standard lot (default 10 USD for FX)
        contract_size      — units per lot (default 100k for standard FX lot)
        """
        if sl_pips <= 0 or account_balance <= 0:
            return RISK_CFG.lot_min

        risk_pct  = risk_pct if risk_pct is not None else RISK_CFG.max_risk_per_trade_pct
        risk_pct  = min(risk_pct, RISK_CFG.max_risk_per_trade_pct)
        risk_dollars = account_balance * (risk_pct / 100.0)
        per_lot_loss = sl_pips * pip_value_per_lot
        lots = risk_dollars / max(per_lot_loss, 1e-9)
        return max(RISK_CFG.lot_min, min(lots, RISK_CFG.lot_max))
