"""RiskEngine: converts a Signal into an approved exposure (spec sections 30-35, 40, 44).

Halts tracked here:
* daily loss halt       -- flat and no new trades for the rest of the session (section 33)
* drawdown halt         -- flat and no trading until a retrain is accepted (section 34)
* fractional-edge halt  -- FRACTIONAL_EDGE_NOT_DETECTED after repeated ablation failures (section 40)
* data halt             -- set by the bot when the DataValidator rejects input (section 35)
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional

from ..types import BotState, FeatureVector, PortfolioSnapshot, RiskDecision, Signal
from .limits import apply_position_rules, stop_triggered


class RiskEngine:
    def __init__(self, max_absolute_exposure: float = 1.0, maximum_holding_bars: int = 12,
                 stop_sigma_multiple: float = 4.0, daily_loss_limit: float = 0.025, drawdown_halt: float = 0.10,
                 rebalance_threshold: float = 0.15, horizon: int = 4, ablation_failures_to_halt: int = 3,
                 halt_recovery_bars: int = 13):
        self.max_abs = float(max_absolute_exposure)
        self.max_holding = int(maximum_holding_bars)
        self.stop_multiple = float(stop_sigma_multiple)
        self.daily_loss_limit = float(daily_loss_limit)
        self.drawdown_limit = float(drawdown_halt)
        self.rebalance_threshold = float(rebalance_threshold)
        self.horizon = int(horizon)
        self.ablation_failures_to_halt = int(ablation_failures_to_halt)
        self.halt_recovery_bars = int(halt_recovery_bars)
        self.clean_bars_since_halt = 0
        # halt state
        self.daily_halt_date: Optional[date] = None
        self.drawdown_halted = False
        self.ablation_halted = False
        self.data_halted = False
        self.ablation_failures = 0
        self.events: list[dict] = []

    @classmethod
    def from_config(cls, cfg) -> "RiskEngine":
        r = cfg.risk
        return cls(r.max_absolute_exposure, r.maximum_holding_bars, r.stop_sigma_multiple, r.daily_loss_limit,
                   r.drawdown_halt, cfg.signal.rebalance_threshold, cfg.prediction.horizon_bars,
                   cfg.training.ablation.consecutive_failures_to_halt, cfg.data.halt_recovery_bars)

    # ---------------------------------------------------------------- halts
    def _event(self, kind: str, **info) -> None:
        self.events.append({"event": kind, **info})

    def set_data_halt(self, halted: bool, reason: str = "") -> None:
        """Arm (or re-arm) / clear the data halt.  Every (re)arming restarts the recovery count."""
        if halted:
            if not self.data_halted:
                self._event("DATA_HALT", reason=reason)
            self.clean_bars_since_halt = 0
        elif self.data_halted:
            self._event("DATA_HALT_CLEARED", clean_bars=self.clean_bars_since_halt)
            self.clean_bars_since_halt = 0
        self.data_halted = halted

    def note_clean_bar(self) -> bool:
        """Count a clean bar while halted; returns True when the halt clears (section 35 recovery)."""
        if not self.data_halted:
            return False
        self.clean_bars_since_halt += 1
        if self.clean_bars_since_halt >= self.halt_recovery_bars:
            self.set_data_halt(False)
            return True
        return False

    def record_retrain(self, accepted: bool, delta_score: float) -> bool:
        """Update drawdown / ablation halts after a retraining cycle (sections 34, 40).

        Returns True when a drawdown halt was lifted, so the caller can re-base the drawdown
        reference (otherwise the still-depressed equity would re-arm the halt on the next bar).
        """
        cleared_drawdown = False
        if math.isfinite(delta_score) and delta_score > 0:
            self.ablation_failures = 0
            if self.ablation_halted:
                self._event("FRACTIONAL_EDGE_RESTORED", delta_score=delta_score)
            self.ablation_halted = False
        else:
            self.ablation_failures += 1
            if self.ablation_failures >= self.ablation_failures_to_halt and not self.ablation_halted:
                self.ablation_halted = True
                self._event("FRACTIONAL_EDGE_NOT_DETECTED", consecutive_failures=self.ablation_failures)
        if accepted and self.drawdown_halted:
            self.drawdown_halted = False
            cleared_drawdown = True
            self._event("DRAWDOWN_HALT_CLEARED")
        return cleared_drawdown

    @property
    def halted(self) -> bool:
        return self.drawdown_halted or self.ablation_halted or self.data_halted or self.daily_halt_date is not None

    def halt_reason(self, session_date: date) -> str:
        reasons = []
        if self.data_halted:
            reasons.append("DATA_HALTED")
        if self.drawdown_halted:
            reasons.append("DRAWDOWN_HALT")
        if self.ablation_halted:
            reasons.append("FRACTIONAL_EDGE_NOT_DETECTED")
        if self.daily_halt_date == session_date:
            reasons.append("DAILY_RISK_HALT")
        return ",".join(reasons)

    def state_for(self, exposure: float) -> BotState:
        if self.data_halted:
            return BotState.DATA_HALTED
        if self.drawdown_halted or self.ablation_halted or self.daily_halt_date is not None:
            return BotState.RISK_HALTED
        return BotState.POSITIONED if exposure != 0.0 else BotState.READY

    # ------------------------------------------------------------- evaluate
    def evaluate(self, signal: Signal, portfolio: PortfolioSnapshot, market_state: FeatureVector,
                 session_date: date) -> RiskDecision:
        proposed = float(max(-self.max_abs, min(self.max_abs, signal.target_exposure)))
        current = portfolio.exposure
        vm = signal.volatility_multiplier
        # Daily halt expires with the session.
        if self.daily_halt_date is not None and self.daily_halt_date != session_date:
            self.daily_halt_date = None
            self._event("DAILY_RISK_HALT_CLEARED", session=str(session_date))
        daily_status = "OK"
        if portfolio.daily_return < -self.daily_loss_limit and self.daily_halt_date != session_date:
            self.daily_halt_date = session_date
            self._event("DAILY_RISK_HALT", daily_return=portfolio.daily_return, session=str(session_date))
        if self.daily_halt_date == session_date:
            daily_status = "DAILY_RISK_HALT"
        dd_status = "OK"
        if portfolio.drawdown < -self.drawdown_limit and not self.drawdown_halted:
            self.drawdown_halted = True
            self._event("DRAWDOWN_HALT", drawdown=portfolio.drawdown)
        if self.drawdown_halted:
            dd_status = "DRAWDOWN_HALT"
        if self.ablation_halted or self.data_halted or self.drawdown_halted or daily_status != "OK":
            reason = self.halt_reason(session_date)
            return RiskDecision(proposed, 0.0, vm, daily_status, dd_status, "OK", reason, "OK",
                                self.state_for(0.0).value, False)
        stop_hit = False
        if portfolio.units != 0 and portfolio.entry_sigma is not None:
            stop_hit = stop_triggered(portfolio.position_return, portfolio.entry_sigma, self.horizon, self.stop_multiple)
        rule = apply_position_rules(proposed, current, portfolio.holding_bars, self.max_holding, stop_hit,
                                    self.rebalance_threshold)
        if rule.reason == "TURNOVER_SUPPRESSED":
            approved = float(current)      # maintain the existing position exactly (section 30): no drift trimming
        else:
            approved = float(max(-self.max_abs, min(self.max_abs, rule.exposure)))
        if rule.stop_status == "TRIGGERED":
            self._event("STOP_LOSS", position_return=portfolio.position_return, entry_sigma=portfolio.entry_sigma)
        return RiskDecision(proposed, approved, vm, daily_status, dd_status, rule.max_holding_status, rule.reason,
                            rule.stop_status, self.state_for(approved).value, rule.new_entry)


__all__ = ["RiskEngine"]
