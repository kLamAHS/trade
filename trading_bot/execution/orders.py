"""Order construction and the next-bar order queue (spec section 42)."""

from __future__ import annotations

import itertools
import math
from datetime import datetime
from typing import Optional

from ..types import Order

_counter = itertools.count(1)


class OrderBuilder:
    def __init__(self, min_order_notional: float = 1.0):
        self.min_order_notional = float(min_order_notional)

    def build(self, instrument: str, signal_timestamp: datetime, current_units: float, current_exposure: float,
              target_exposure: float, equity: float, estimated_price: float, reason: str = "rebalance",
              new_entry: bool = False, entry_sigma: float = math.nan) -> Optional[Order]:
        if not (math.isfinite(estimated_price) and estimated_price > 0 and math.isfinite(equity) and equity > 0):
            return None
        target_units = target_exposure * equity / estimated_price
        delta_units = target_units - current_units
        if abs(delta_units) * estimated_price < self.min_order_notional:
            return None
        side = "buy" if delta_units > 0 else "sell"
        return Order(instrument=instrument, signal_timestamp=signal_timestamp, side=side, units=abs(delta_units),
                     target_exposure=float(target_exposure), current_exposure=float(current_exposure),
                     estimated_price=float(estimated_price), reason=reason, order_id=f"o{next(_counter):08d}",
                     new_entry=new_entry, entry_sigma=float(entry_sigma))


class OrderQueue:
    def __init__(self):
        self._pending: list[Order] = []

    def push(self, order: Order) -> None:
        self._pending.append(order)

    def pop_all(self) -> list[Order]:
        out, self._pending = self._pending, []
        return out

    def peek(self) -> list[Order]:
        return list(self._pending)

    def __len__(self) -> int:
        return len(self._pending)

    def clear(self) -> None:
        self._pending.clear()


__all__ = ["OrderBuilder", "OrderQueue"]
