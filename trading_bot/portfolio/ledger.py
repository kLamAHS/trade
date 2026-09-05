"""PortfolioLedger: cash, position, marks, P&L, drawdown and holding-time bookkeeping.

Conventions
* Spread and slippage are embedded in the fill price; commission is charged to cash
  explicitly.  Per-trade *net* P&L is therefore ``realized_pnl - commission``; the
  ``costs`` field of a trade is an informational decomposition only.
* ``session_start_equity`` is the equity at the previous session's close, so overnight
  gaps and the opening bar count toward the daily loss limit (spec section 33).
* ``holding_bars`` counts marks while positioned; a max-holding re-entry in the same
  direction (spec section 31) is recorded through :meth:`reenter`, which closes the
  trade record and restarts the holding clock and stop reference (section 32).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from ..types import Fill, PortfolioSnapshot


@dataclass
class TradeRecord:
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: int
    entry_price: float
    exit_price: Optional[float]
    max_units: float
    realized_pnl: float
    costs: float               # spread + slippage + commission (informational)
    bars_held: int
    entry_sigma: float
    fills: int = 0
    commission: float = 0.0
    exit_reason: str = ""

    @property
    def net_pnl(self) -> float:
        """Realised P&L net of all costs (spread/slippage are already inside the fill prices)."""
        return self.realized_pnl - self.commission

    def to_dict(self) -> dict:
        return {"entry_time": self.entry_time.isoformat(), "exit_time": self.exit_time.isoformat() if self.exit_time else None,
                "direction": self.direction, "entry_price": self.entry_price, "exit_price": self.exit_price,
                "max_units": self.max_units, "realized_pnl": self.realized_pnl, "net_pnl": self.net_pnl,
                "costs": self.costs, "commission": self.commission, "bars_held": self.bars_held,
                "entry_sigma": self.entry_sigma, "fills": self.fills, "exit_reason": self.exit_reason}


class PortfolioLedger:
    def __init__(self, initial_capital: float, instrument: str):
        self.initial_capital = float(initial_capital)
        self.instrument = instrument
        self.cash = float(initial_capital)
        self.units = 0.0
        self.avg_price = math.nan
        self.mark_price = math.nan
        self.mark_time: Optional[datetime] = None
        self.realized_pnl = 0.0
        self.total_costs = 0.0
        self.total_commission = 0.0
        self.total_spread_cost = 0.0
        self.total_slippage_cost = 0.0
        self.equity_peak = float(initial_capital)
        self.session_date: Optional[date] = None
        self.session_start_equity = float(initial_capital)
        self._last_mark_equity = float(initial_capital)
        self.entry_bar_index: Optional[int] = None
        self.entry_price: Optional[float] = None
        self.entry_sigma: Optional[float] = None
        self.entry_direction = 0
        self.entry_time: Optional[datetime] = None
        self.holding_bars = 0
        self.bar_index = -1
        self.fills: list[Fill] = []
        self.trades: list[TradeRecord] = []
        self._open_trade: Optional[TradeRecord] = None
        self.turnover_notional = 0.0
        self.equity_history: list[tuple[datetime, float, float]] = []   # (time, equity, exposure)

    # --------------------------------------------------------------- state
    @property
    def equity(self) -> float:
        px = self.mark_price if math.isfinite(self.mark_price) else (self.avg_price if math.isfinite(self.avg_price) else 0.0)
        return self.cash + self.units * px

    @property
    def exposure(self) -> float:
        eq = self.equity
        if eq <= 0 or not math.isfinite(self.mark_price):
            return 0.0
        return self.units * self.mark_price / eq

    @property
    def unrealized_pnl(self) -> float:
        if self.units == 0 or not math.isfinite(self.avg_price) or not math.isfinite(self.mark_price):
            return 0.0
        return self.units * (self.mark_price - self.avg_price)

    @property
    def drawdown(self) -> float:
        return (self.equity - self.equity_peak) / self.equity_peak if self.equity_peak > 0 else 0.0

    @property
    def daily_return(self) -> float:
        return (self.equity - self.session_start_equity) / self.session_start_equity if self.session_start_equity > 0 else 0.0

    @property
    def position_return(self) -> float:
        if self.units == 0 or self.entry_price is None or not math.isfinite(self.mark_price):
            return 0.0
        return self.entry_direction * math.log(self.mark_price / self.entry_price)

    def state(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            timestamp=self.mark_time, cash=self.cash, units=self.units, mark_price=self.mark_price, equity=self.equity,
            exposure=self.exposure, equity_peak=self.equity_peak, drawdown=self.drawdown,
            session_start_equity=self.session_start_equity, daily_return=self.daily_return,
            realized_pnl=self.realized_pnl, unrealized_pnl=self.unrealized_pnl, entry_bar_index=self.entry_bar_index,
            entry_price=self.entry_price, entry_sigma=self.entry_sigma, holding_bars=self.holding_bars,
            position_return=self.position_return)

    # --------------------------------------------------------------- marks
    def mark(self, timestamp: datetime, price: float, bar_index: int, session_date: date) -> None:
        if session_date != self.session_date:
            # Daily P&L is measured from the previous session's closing equity (initial capital before the
            # first mark), so overnight gaps and fills at the open count toward the daily limit.
            self.session_date = session_date
            self.session_start_equity = self._last_mark_equity
        self.mark_price = float(price)
        self.mark_time = timestamp
        self.bar_index = bar_index
        if self.units != 0:
            self.holding_bars += 1
        eq = self.equity
        if eq > self.equity_peak:
            self.equity_peak = eq
        self._last_mark_equity = eq
        self.equity_history.append((timestamp, eq, self.exposure))

    def rebase_peak(self) -> None:
        """Restart the drawdown measurement from the current equity (after a drawdown halt is lifted)."""
        self.equity_peak = max(self.equity, 1e-12)

    # --------------------------------------------------------------- fills
    def apply(self, fill: Fill) -> None:
        signed = fill.signed_units
        cost = fill.spread_cost + fill.slippage_cost + fill.commission
        self.total_costs += cost
        self.total_commission += fill.commission
        self.total_spread_cost += fill.spread_cost
        self.total_slippage_cost += fill.slippage_cost
        self.turnover_notional += fill.notional
        self.cash -= signed * fill.fill_price
        self.cash -= fill.commission      # spread/slippage are embedded in fill_price; commission is explicit
        prev_units = self.units
        new_units = prev_units + signed
        if prev_units == 0 or (prev_units > 0) == (signed > 0):
            total = abs(prev_units) + abs(signed)
            self.avg_price = ((abs(prev_units) * (self.avg_price if math.isfinite(self.avg_price) else fill.fill_price))
                              + abs(signed) * fill.fill_price) / total
        else:
            closed = min(abs(prev_units), abs(signed))
            direction = 1 if prev_units > 0 else -1
            pnl = closed * direction * (fill.fill_price - self.avg_price)
            self.realized_pnl += pnl
            if self._open_trade is not None:
                self._open_trade.realized_pnl += pnl
            if abs(signed) > abs(prev_units):
                self.avg_price = fill.fill_price          # flipped: remainder opens at the fill price
            elif abs(new_units) < 1e-9:
                self.avg_price = math.nan
        self.units = new_units
        if abs(self.units) < 1e-9:
            self.units = 0.0
        self.fills.append(fill)
        self._update_position_bookkeeping(fill, prev_units)
        if not math.isfinite(self.mark_price):
            self.mark_price = fill.fill_price

    def _update_position_bookkeeping(self, fill: Fill, prev_units: float) -> None:
        cost = fill.spread_cost + fill.slippage_cost + fill.commission
        flat_now = self.units == 0.0
        flipped = prev_units != 0 and not flat_now and (prev_units > 0) != (self.units > 0)
        opened = prev_units == 0 and not flat_now
        if self._open_trade is not None:
            # the old trade never held more than its pre-fill size when this fill closes or flips it
            size_for_old = abs(prev_units) if (flat_now or flipped) else abs(self.units)
            self._open_trade.max_units = max(self._open_trade.max_units, size_for_old)
            if flipped:
                # closing part of the turnover belongs to the old trade, the opening part to the new one
                frac = abs(prev_units) / fill.units if fill.units else 1.0
                self._open_trade.costs += cost * frac
                self._open_trade.commission += fill.commission * frac
                cost_new = cost * (1.0 - frac)
                commission_new = fill.commission * (1.0 - frac)
            else:
                self._open_trade.costs += cost
                self._open_trade.commission += fill.commission
                cost_new = commission_new = 0.0
            self._open_trade.fills += 1
        else:
            cost_new, commission_new = cost, fill.commission
        if flat_now or flipped:
            self._close_trade(fill.fill_timestamp, fill.fill_price, "flip" if flipped else "flat")
        if opened or flipped:
            self._open_position(fill.fill_timestamp, fill.fill_price, fill.entry_sigma, self.bar_index, cost_new,
                                commission_new)

    def _close_trade(self, when: datetime, price: float, reason: str) -> None:
        if self._open_trade is not None:
            self._open_trade.exit_time = when
            self._open_trade.exit_price = price
            self._open_trade.bars_held = self.holding_bars
            self._open_trade.exit_reason = reason
            self.trades.append(self._open_trade)
            self._open_trade = None
        self.entry_bar_index = None
        self.entry_price = None
        self.entry_sigma = None
        self.entry_direction = 0
        self.entry_time = None
        self.holding_bars = 0

    def _open_position(self, when: datetime, price: float, sigma: float, bar_index: int, cost: float,
                       commission: float) -> None:
        self.entry_bar_index = bar_index
        self.entry_price = price
        self.entry_sigma = sigma if (sigma is not None and math.isfinite(sigma)) else None
        self.entry_direction = 1 if self.units > 0 else -1
        self.entry_time = when
        self.holding_bars = 0
        self._open_trade = TradeRecord(when, None, self.entry_direction, price, None, abs(self.units), 0.0, cost, 0,
                                       sigma if sigma is not None else math.nan, 1, commission)

    def reenter(self, when: datetime, price: float, sigma: float, bar_index: int) -> None:
        """Maximum-holding re-entry in the same direction (section 31): the existing position is treated
        as closed and re-opened at the current mark, restarting the holding clock and the stop reference."""
        if self.units == 0:
            return
        direction = 1 if self.units > 0 else -1
        # realise the open trade's mark-to-market so the new trade starts from zero
        if self._open_trade is not None and math.isfinite(self.avg_price):
            mtm = self.units * (price - self.avg_price)
            self._open_trade.realized_pnl += mtm
            self.realized_pnl += mtm
            self.avg_price = price
        self._close_trade(when, price, "max_holding_reentry")
        self._open_position(when, price, sigma, bar_index, 0.0, 0.0)
        self.entry_direction = direction
        if self._open_trade is not None:
            self._open_trade.fills = 0

    def summary(self) -> dict:
        return {"equity": self.equity, "cash": self.cash, "units": self.units, "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl, "total_costs": self.total_costs,
                "commission": self.total_commission, "spread_cost": self.total_spread_cost,
                "slippage_cost": self.total_slippage_cost, "turnover_notional": self.turnover_notional,
                "n_fills": len(self.fills), "n_trades": len(self.trades), "equity_peak": self.equity_peak,
                "drawdown": self.drawdown, "session_start_equity": self.session_start_equity}


__all__ = ["PortfolioLedger", "TradeRecord"]
