"""Execution engine: fill simulation (spec section 43), state-consistent order
handling (section 44) and an optional Alpaca paper-trading mirror.

Fill model (the ledger's source of truth in every mode):
    buy : Open_{t+1} + Spread/2 + Slippage
    sell: Open_{t+1} - Spread/2 - Slippage

Temporal rule: a fill may only use a price observed at or after the decision.  In a
bar backtest the decision is stamped at the bar close, which is the next bar's open
print, so the next open is the first tradable price.  In live paper mode the
decision is made seconds after the close (bars arrive with a delay), so the next
bar's open print already predates it: the fill then uses the NBBO quote observed at
decision time (buy at ask, sell at bid, plus slippage), and if no such quote exists
the order is deferred to the following bar's open.  It is never filled at a price
that existed before the signal.

Live trading is not supported by this application: the broker adapter refuses any
configuration other than the Alpaca *paper* endpoint.

Alpaca mirror: when a broker is attached, every order is submitted to the paper
account the moment it is queued (in live mode that instant is the open of bar
t+1).  The broker's response and later fill status are recorded on the Fill as
an annotation (``Fill.mirror``) and as audit events; they never replace the
simulated fill, so backtest and paper accounting stay identical.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from ..types import Bar, BotState, CostEstimate, FeatureVector, Fill, Order
from .cost_model import CostModel
from .orders import OrderBuilder, OrderQueue


class FillDeferred(Exception):
    """No price observed at/after the decision is available on this bar; fill on a later bar."""


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
        price = ref + half_spread_px + slip_px if order.side == "buy" else ref - half_spread_px - slip_px
        return price, half_spread_px, slip_px

    def simulate_fill(self, order: Order, exec_bar: Bar, signal_bar: Bar | None = None,
                      mirror: Optional[dict[str, Any]] = None) -> Fill:
        if exec_bar.close_time <= order.signal_timestamp:
            raise ValueError("execution bar ended before the signal timestamp")
        if exec_bar.timestamp >= order.signal_timestamp:
            # The open print occurs at/after the decision: the standard next-open fill.
            price, half_spread_px, slip_px = self.fill_price(order, exec_bar, signal_bar)
            reference, fill_time, source = exec_bar.open, exec_bar.timestamp, "next_open"
        else:
            # The bar opened before the decision was made; its open is not an available price.
            quote_ok = (signal_bar is not None and signal_bar.bid is not None and signal_bar.ask is not None
                        and signal_bar.quote_timestamp is not None
                        and signal_bar.quote_timestamp >= order.signal_timestamp - timedelta(seconds=1))
            if not quote_ok:
                raise FillDeferred("no price observed at or after the decision on this bar")
            range_rel = (signal_bar.high - signal_bar.low) / signal_bar.close
            mid = 0.5 * (signal_bar.bid + signal_bar.ask)
            slip_px = self.cost_model.slippage_per_side(range_rel) * mid
            half_spread_px = 0.5 * (signal_bar.ask - signal_bar.bid)
            price = (signal_bar.ask + slip_px) if order.side == "buy" else (signal_bar.bid - slip_px)
            reference, fill_time, source = mid, max(signal_bar.quote_timestamp, order.signal_timestamp), "quote"
        notional = order.units * price
        return Fill(order_id=order.order_id, instrument=order.instrument, signal_timestamp=order.signal_timestamp,
                    fill_timestamp=fill_time, side=order.side, units=order.units, reference_price=reference,
                    fill_price=price, spread_cost=half_spread_px * order.units, slippage_cost=slip_px * order.units,
                    commission=self.cost_model.commission_per_side * notional, source="simulator",
                    new_entry=order.new_entry, entry_sigma=order.entry_sigma, target_exposure=order.target_exposure,
                    mirror=mirror, price_source=source)


class LiveTradingNotSupported(RuntimeError):
    """Raised whenever anything other than the Alpaca paper endpoint would be used."""


def paper_trading_client(api_key: str, secret_key: str):
    """The only way this application constructs an Alpaca TradingClient: paper endpoint, verified."""
    from alpaca.trading.client import TradingClient

    if not api_key or not secret_key:
        raise RuntimeError("Alpaca credentials missing: set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
    client = TradingClient(api_key, secret_key, paper=True)
    base = str(getattr(client, "_base_url", "") or "")
    if base and "paper" not in base.lower():
        raise LiveTradingNotSupported(f"trading client is not on the paper endpoint ({base})")
    return client


class AlpacaPaperBroker:
    """Mirrors exposure changes as whole-share market orders to an Alpaca *paper* account.

    Live trading is not supported: ``paper`` must be True and the client is verified to
    target the paper endpoint.  The internal ledger remains the source of truth; broker
    responses are annotations.  Quantities are rounded to whole shares because Alpaca
    does not accept fractional short sales.
    """

    FINAL = ("filled", "canceled", "cancelled", "expired", "rejected", "replaced", "stopped", "suspended")

    def __init__(self, api_key: str | None = None, secret_key: str | None = None, paper: bool = True,
                 trading_client=None):
        import os

        if paper is not True:
            raise LiveTradingNotSupported("live trading is not supported by this application (paper must be True)")
        self.api_key = api_key or os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        self.paper = True
        self._client = trading_client

    @property
    def client(self):
        if self._client is None:
            self._client = paper_trading_client(self.api_key, self.secret_key)
        return self._client

    @staticmethod
    def _summary(o) -> dict[str, Any]:
        status = str(getattr(o, "status", "")).lower().split(".")[-1]
        avg = getattr(o, "filled_avg_price", None)
        qty = getattr(o, "filled_qty", None)
        return {"id": str(getattr(o, "id", "")), "status": status,
                "filled_qty": float(qty) if qty not in (None, "") else 0.0,
                "filled_avg_price": float(avg) if avg not in (None, "") else None,
                "final": status in AlpacaPaperBroker.FINAL}

    def _submit_leg(self, symbol: str, side: str, qty: int, client_order_id: str) -> dict[str, Any]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                                 time_in_force=TimeInForce.DAY, client_order_id=client_order_id[:48])
        try:
            resp = self.client.submit_order(req)
        except Exception as exc:
            return {"id": None, "status": "error", "reason": str(exc), "final": True, "qty": qty}
        out = self._summary(resp)
        out["qty"] = qty
        return out

    def submit(self, order: Order, wait_close_seconds: float = 5.0) -> dict[str, Any]:
        """Submit immediately and return the broker's acknowledgement (no blocking poll).

        Alpaca rejects a single order that takes a position through zero, so a flip is sent as a
        closing leg followed by an opening leg for the remainder.
        """
        qty = int(round(order.units))
        if qty <= 0:
            return {"id": None, "status": "skipped", "reason": "rounds to zero shares", "final": True}
        signed = qty if order.side == "buy" else -qty
        position = self.position_qty(order.instrument) or 0.0
        crosses = position != 0 and (position > 0) != (signed > 0) and abs(signed) > abs(position)
        submitted_at = datetime.now(timezone.utc).isoformat()
        if not crosses:
            out = self._submit_leg(order.instrument, order.side, qty, order.order_id)
            out["submitted_at"] = submitted_at
            return out
        close_qty = int(round(abs(position)))
        open_qty = qty - close_qty
        close_leg = self._submit_leg(order.instrument, order.side, close_qty, order.order_id + "c")
        if close_leg.get("id"):
            deadline = time.time() + wait_close_seconds
            while time.time() < deadline and not close_leg.get("final"):
                time.sleep(0.5)
                close_leg = {**close_leg, **self.check(close_leg["id"])}
        open_leg = self._submit_leg(order.instrument, order.side, open_qty, order.order_id + "o") if open_qty > 0 else None
        out = dict(open_leg or close_leg)
        out.update({"submitted_at": submitted_at, "qty": qty, "flip": True, "close_leg": close_leg, "open_leg": open_leg})
        return out

    def cancel(self, broker_order_id: str) -> dict[str, Any]:
        try:
            self.client.cancel_order_by_id(broker_order_id)
        except Exception as exc:
            return {"id": broker_order_id, "status": "error", "reason": str(exc), "final": False}
        return self.check(broker_order_id)

    def check(self, broker_order_id: str) -> dict[str, Any]:
        try:
            return self._summary(self.client.get_order_by_id(broker_order_id))
        except Exception as exc:
            return {"id": broker_order_id, "status": "error", "reason": str(exc), "final": False}

    def position_qty(self, symbol: str) -> Optional[float]:
        try:
            pos = self.client.get_open_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0 if self._client is not None else None

    def account_equity(self) -> Optional[float]:
        try:
            return float(self.client.get_account().equity)
        except Exception:  # pragma: no cover
            return None


class ExecutionEngine:
    """Builds, queues and fills orders; enforces state consistency (section 44)."""

    def __init__(self, cost_model: CostModel, order_builder: OrderBuilder | None = None,
                 slippage_reference: str = "execution_bar", broker: AlpacaPaperBroker | None = None,
                 live_fill_source: str = "quote"):
        if live_fill_source not in ("quote", "broker"):
            raise ValueError("execution.live_fill_source must be 'quote' or 'broker'")
        self.cost_model = cost_model
        self.simulator = ExecutionSimulator(cost_model, slippage_reference)
        self.builder = order_builder or OrderBuilder()
        self.queue = OrderQueue()
        self.broker = broker
        self.live_fill_source = live_fill_source
        self.available = True
        self.rejected: list[tuple[Order, str]] = []
        self.mirror_acks: dict[str, dict[str, Any]] = {}      # order_id -> broker acknowledgement
        self.pending_mirrors: dict[str, str] = {}             # order_id -> broker order id awaiting a final status
        self.events: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, cfg, broker: AlpacaPaperBroker | None = None, id_prefix: str | None = None) -> "ExecutionEngine":
        e = cfg.execution
        return cls(CostModel.from_config(cfg), OrderBuilder(e.min_order_notional, id_prefix), e.slippage_reference, broker,
                   str(e.get("live_fill_source", "quote")))

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
        if state in (BotState.RISK_HALTED, BotState.DATA_HALTED) and order.target_exposure != 0.0:
            return False, f"only flattening orders are allowed while {state.value}"
        return True, "OK"

    def queue_for_next_bar(self, order: Order) -> None:
        """Queue for the next bar's open; with a broker attached the mirror order is submitted now."""
        self.queue.push(order)
        if self.broker is not None:
            ack = self.broker.submit(order)
            self.mirror_acks[order.order_id] = ack
            self.events.append({"event": "ORDER_MIRROR", "order_id": order.order_id, **ack})
            if ack.get("id") and not ack.get("final"):
                self.pending_mirrors[order.order_id] = ack["id"]

    def pending_orders(self) -> list[Order]:
        return self.queue.pop_all()

    def cancel_pending(self) -> list[Order]:
        """Drop queued orders; mirrored orders that are not final are cancelled at the broker and any
        quantity that already filled there is reported so the drift is visible in the audit trail."""
        orders = self.queue.pop_all()
        for order in orders:
            ack = self.mirror_acks.pop(order.order_id, None)
            broker_id = self.pending_mirrors.pop(order.order_id, None)
            if self.broker is None or ack is None:
                continue
            if broker_id is not None:
                status = self.broker.cancel(broker_id)
                self.events.append({"event": "ORDER_MIRROR_CANCELLED", "order_id": order.order_id, **status})
                filled = status.get("filled_qty") or 0.0
            else:
                filled = ack.get("filled_qty") or 0.0
            if filled:
                self.events.append({"event": "MIRROR_DRIFT_RISK", "order_id": order.order_id, "filled_qty": filled,
                                    "note": "broker filled part of an order the ledger dropped; see reconciliation"})
        return orders

    def has_pending(self) -> bool:
        return len(self.queue) > 0

    # ---------------------------------------------------------------- fills
    def simulate_fill(self, order: Order, bar: Bar, signal_bar: Bar | None = None) -> Fill:
        if not self.available:
            raise RuntimeError("execution simulator unavailable")
        mirror = self.mirror_acks.pop(order.order_id, None)
        if mirror is not None and order.order_id in self.pending_mirrors and self.broker is not None:
            latest = self.broker.check(self.pending_mirrors[order.order_id])
            mirror = {**mirror, **{k: v for k, v in latest.items() if v is not None}}
            if latest.get("final"):
                self.pending_mirrors.pop(order.order_id, None)
                self.events.append({"event": "ORDER_MIRROR_FINAL", "order_id": order.order_id, **latest})
        broker_px = (mirror or {}).get("filled_avg_price") if (mirror and mirror.get("status") == "filled") else None
        try:
            fill = self.simulator.simulate_fill(order, bar, signal_bar, mirror)
        except FillDeferred:
            if self.live_fill_source == "broker" and broker_px:
                fill = None
            else:
                raise
        if self.live_fill_source == "broker" and broker_px:
            # Paper mode with the broker's actual fill as the ledger price (observed after the decision by
            # construction: the mirror order was submitted at decision time).
            px = float(broker_px)
            units = float(mirror.get("filled_qty") or order.units) if mirror.get("qty") else order.units
            units = min(units, order.units) if units > 0 else order.units
            ref = fill.reference_price if fill is not None else px
            adverse = max(0.0, (px - ref) if order.side == "buy" else (ref - px))
            return Fill(order_id=order.order_id, instrument=order.instrument, signal_timestamp=order.signal_timestamp,
                        fill_timestamp=max(order.signal_timestamp, bar.timestamp), side=order.side, units=units,
                        reference_price=ref, fill_price=px, spread_cost=adverse * units, slippage_cost=0.0,
                        commission=self.cost_model.commission_per_side * px * units, source="broker",
                        new_entry=order.new_entry, entry_sigma=order.entry_sigma, target_exposure=order.target_exposure,
                        mirror=mirror, price_source="broker")
        return fill

    def poll_mirrors(self) -> None:
        """Refresh broker statuses for mirrored orders that were not final at fill time."""
        if self.broker is None or not self.pending_mirrors:
            return
        for order_id, broker_id in list(self.pending_mirrors.items()):
            latest = self.broker.check(broker_id)
            if latest.get("final"):
                self.pending_mirrors.pop(order_id, None)
                self.events.append({"event": "ORDER_MIRROR_FINAL", "order_id": order_id, **latest})

    def reconcile_mirror(self, instrument: str, ledger_units: float, tolerance: float = 1.0) -> Optional[dict[str, Any]]:
        """Compare the broker's position with the ledger; returns a drift record when they differ."""
        if self.broker is None:
            return None
        qty = self.broker.position_qty(instrument)
        if qty is None:
            return None
        return {"broker_qty": qty, "ledger_units": ledger_units, "drift": qty - ledger_units,
                "flagged": abs(qty - ledger_units) > tolerance}


__all__ = ["ExecutionSimulator", "ExecutionEngine", "AlpacaPaperBroker", "FillDeferred", "LiveTradingNotSupported",
           "paper_trading_client"]
