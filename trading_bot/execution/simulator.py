"""Execution engine: fill simulation (spec section 43), state-consistent order
handling (section 44) and an optional Alpaca paper-trading mirror.

Fill model:
    buy : Open_{t+1} + Spread/2 + Slippage
    sell: Open_{t+1} - Spread/2 - Slippage
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Optional

from ..types import Bar, BotState, CostEstimate, FeatureVector, Fill, Order
from .cost_model import CostModel
from .orders import OrderBuilder, OrderQueue


class ExecutionSimulator:
    def __init__(self, cost_model: CostModel, slippage_reference: str = "execution_bar"):
        self.cost_model = cost_model
        self.slippage_reference = slippage_reference

    def fill_price(self, order: Order, exec_bar: Bar, signal_bar: Bar | None = None) -> tuple[float, float, float]:
        ref = exec_bar.open
        spread_rel = exec_bar.relative_spread
        if spread_rel is None and signal_bar is not None:
            spread_rel = signal_bar.relative_spread
        spread = self.cost_model.spread(spread_rel)
        range_bar = exec_bar if (self.slippage_reference == "execution_bar" or signal_bar is None) else signal_bar
        range_rel = (range_bar.high - range_bar.low) / range_bar.close
        slip = self.cost_model.slippage_per_side(range_rel)
        half_spread_px = 0.5 * spread * ref
        slip_px = slip * ref
        if order.side == "buy":
            price = ref + half_spread_px + slip_px
        else:
            price = ref - half_spread_px - slip_px
        return price, half_spread_px, slip_px

    def simulate_fill(self, order: Order, exec_bar: Bar, signal_bar: Bar | None = None) -> Fill:
        if exec_bar.timestamp < order.signal_timestamp:
            raise ValueError("execution bar precedes the signal timestamp")
        price, half_spread_px, slip_px = self.fill_price(order, exec_bar, signal_bar)
        notional = order.units * price
        return Fill(order_id=order.order_id, instrument=order.instrument, signal_timestamp=order.signal_timestamp,
                    fill_timestamp=exec_bar.timestamp, side=order.side, units=order.units, reference_price=exec_bar.open,
                    fill_price=price, spread_cost=half_spread_px * order.units, slippage_cost=slip_px * order.units,
                    commission=self.cost_model.commission_per_side * notional, source="simulator",
                    new_entry=order.new_entry, entry_sigma=order.entry_sigma, target_exposure=order.target_exposure)


class AlpacaPaperBroker:
    """Mirrors exposure changes as market orders to an Alpaca paper account.

    The internal ledger remains the source of truth; when Alpaca reports a fill
    price it replaces the simulated fill price so the audit trail records what
    actually happened in the paper account.
    """

    def __init__(self, api_key: str | None = None, secret_key: str | None = None, paper: bool = True,
                 fill_timeout_seconds: int = 60, trading_client=None):
        import os

        self.api_key = api_key or os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        self.paper = paper
        self.fill_timeout = fill_timeout_seconds
        self._client = trading_client

    @property
    def client(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient

            if not self.api_key or not self.secret_key:
                raise RuntimeError("Alpaca credentials missing: set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
            self._client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        return self._client

    def submit(self, order: Order) -> Optional[dict]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        qty = round(order.units, 6)
        if qty <= 0:
            return None
        req = MarketOrderRequest(symbol=order.instrument, qty=qty,
                                 side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                                 time_in_force=TimeInForce.DAY, client_order_id=order.order_id[:48])
        resp = self.client.submit_order(req)
        deadline = time.time() + self.fill_timeout
        while time.time() < deadline:
            o = self.client.get_order_by_id(resp.id)
            status = str(getattr(o, "status", "")).lower()
            if "filled" in status and getattr(o, "filled_avg_price", None):
                return {"id": str(o.id), "filled_avg_price": float(o.filled_avg_price),
                        "filled_qty": float(o.filled_qty or qty), "status": status}
            if any(s in status for s in ("canceled", "rejected", "expired")):
                return {"id": str(o.id), "status": status}
            time.sleep(1.0)
        return {"id": str(resp.id), "status": "timeout"}

    def account_equity(self) -> Optional[float]:
        try:
            return float(self.client.get_account().equity)
        except Exception:  # pragma: no cover
            return None


class ExecutionEngine:
    """Builds, queues and fills orders; enforces state consistency (section 44)."""

    def __init__(self, cost_model: CostModel, order_builder: OrderBuilder | None = None,
                 slippage_reference: str = "execution_bar", broker: AlpacaPaperBroker | None = None):
        self.cost_model = cost_model
        self.simulator = ExecutionSimulator(cost_model, slippage_reference)
        self.builder = order_builder or OrderBuilder()
        self.queue = OrderQueue()
        self.broker = broker
        self.available = True
        self.rejected: list[tuple[Order, str]] = []

    @classmethod
    def from_config(cls, cfg, broker: AlpacaPaperBroker | None = None) -> "ExecutionEngine":
        e = cfg.execution
        return cls(CostModel.from_config(cfg), OrderBuilder(e.min_order_notional), e.slippage_reference, broker)

    # ---------------------------------------------------------------- costs
    def estimate_cost(self, market_state: FeatureVector) -> CostEstimate:
        return self.cost_model.estimate(market_state.get("range_rel"), market_state.get("spread_rel"))

    def estimate_cost_from_bar(self, bar: Bar) -> CostEstimate:
        return self.cost_model.estimate((bar.high - bar.low) / bar.close, bar.relative_spread)

    # --------------------------------------------------------------- orders
    def build_order(self, instrument: str, signal_timestamp: datetime, current_units: float, current_exposure: float,
                    target_exposure: float, equity: float, estimated_price: float, state: BotState,
                    reason: str = "rebalance", new_entry: bool = False, entry_sigma: float = math.nan) -> Optional[Order]:
        order = self.builder.build(instrument, signal_timestamp, current_units, current_exposure, target_exposure,
                                   equity, estimated_price, reason, new_entry, entry_sigma)
        if order is None:
            return None
        ok, why = self.consistent_with_state(order, state)
        if not ok:
            self.rejected.append((order, why))
            return None
        return order

    @staticmethod
    def consistent_with_state(order: Order, state: BotState) -> tuple[bool, str]:
        if state == BotState.INITIALIZING:
            return False, "orders are not allowed while INITIALIZING"
        increases = abs(order.target_exposure) > abs(order.current_exposure) + 1e-12 or \
            (order.target_exposure != 0 and math.copysign(1, order.target_exposure) != math.copysign(1, order.current_exposure or order.target_exposure))
        if state in (BotState.RISK_HALTED, BotState.DATA_HALTED) and order.target_exposure != 0.0:
            return False, f"only flattening orders are allowed while {state.value}"
        if state in (BotState.RISK_HALTED, BotState.DATA_HALTED) and increases:
            return False, f"exposure increase rejected while {state.value}"
        return True, "OK"

    def queue_for_next_bar(self, order: Order) -> None:
        self.queue.push(order)

    def pending_orders(self) -> list[Order]:
        return self.queue.pop_all()

    def has_pending(self) -> bool:
        return len(self.queue) > 0

    # ---------------------------------------------------------------- fills
    def simulate_fill(self, order: Order, bar: Bar, signal_bar: Bar | None = None) -> Fill:
        if not self.available:
            raise RuntimeError("execution simulator unavailable")
        fill = self.simulator.simulate_fill(order, bar, signal_bar)
        if self.broker is not None:
            try:
                resp = self.broker.submit(order)
            except Exception as exc:  # pragma: no cover - network dependent
                resp = {"status": f"error: {exc}"}
            if resp and resp.get("filled_avg_price"):
                px = float(resp["filled_avg_price"])
                units = float(resp.get("filled_qty", order.units))
                fill = Fill(order_id=fill.order_id, instrument=fill.instrument, signal_timestamp=fill.signal_timestamp,
                            fill_timestamp=datetime.now(timezone.utc), side=fill.side, units=units,
                            reference_price=bar.open, fill_price=px,
                            spread_cost=max(0.0, (px - bar.open) * units if order.side == "buy" else (bar.open - px) * units),
                            slippage_cost=0.0, commission=fill.commission, source="alpaca",
                            new_entry=fill.new_entry, entry_sigma=fill.entry_sigma, target_exposure=fill.target_exposure)
        return fill


__all__ = ["ExecutionSimulator", "ExecutionEngine", "AlpacaPaperBroker"]
