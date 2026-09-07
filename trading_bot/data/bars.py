"""Event-based bar engine (market-state spec sections 5-7).

``MarketEvent`` is the canonical normalised record (trade / quote / bar).  ``BarBuilder`` turns a
stream of events -- or of finer *time* bars, the only granularity Alpaca's bar endpoints provide --
into TIME, VOLUME, DOLLAR or TICK bars that satisfy the bar contract:

    bar_id, symbol, start_time, end_time, available_at, open, high, low, close, volume,
    dollar_volume, trade_count, vwap, duration_seconds

``available_at`` (== ``Bar.latest_source_time``) is the close time of the last contributing event,
never earlier.  Event bars never straddle a session boundary: an open bucket is flushed at the
session close even if its target is not reached (a partial bar, flagged by ``bar_id`` suffix).
Because features on event bars see variable elapsed time, the feature engine exposes
``duration_seconds`` and per-unit-time normalisations (section 6.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Iterator, Optional

from ..types import Bar
from .calendar import SessionCalendar

BAR_KINDS = ("time", "volume", "dollar", "tick")


@dataclass(frozen=True)
class MarketEvent:
    symbol: str
    exchange_timestamp: datetime
    received_timestamp: datetime
    sequence_id: int
    event_type: str                       # TRADE | QUOTE | BAR
    trade_price: Optional[float] = None
    trade_size: Optional[float] = None
    bid_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_price: Optional[float] = None
    ask_size: Optional[float] = None
    source: str = ""
    # BAR events carry the whole OHLCV of a finer bar so that event bars can be built from bar feeds.
    bar: Optional[Bar] = None

    @property
    def available_at(self) -> datetime:
        return max(self.received_timestamp, self.exchange_timestamp)


def events_from_bars(bars: Iterable[Bar], source: str = "bars") -> Iterator[MarketEvent]:
    for k, b in enumerate(bars):
        yield MarketEvent(b.instrument, b.timestamp, b.latest_source_time, k, "BAR", trade_price=b.close, trade_size=b.volume,
                          bid_price=b.bid, bid_size=b.bid_size, ask_price=b.ask, ask_size=b.ask_size, source=source, bar=b)


class BarBuilder:
    """Accumulates events into bars of the requested kind.  ``target`` is shares (volume), notional
    (dollar), trades / fine bars (tick) or minutes (time)."""

    def __init__(self, kind: str, target: float, calendar: SessionCalendar | None = None, symbol: str = ""):
        kind = str(kind).lower()
        if kind not in BAR_KINDS:
            raise ValueError(f"bar kind must be one of {BAR_KINDS}, got {kind!r}")
        self.kind = kind
        self.target = float(target)
        self.calendar = calendar or SessionCalendar()
        self.symbol = symbol
        self._reset()
        self._count = 0

    def _reset(self) -> None:
        self._open = self._high = self._low = self._close = math.nan
        self._volume = 0.0
        self._dollar = 0.0
        self._trades = 0
        self._start: Optional[datetime] = None
        self._end: Optional[datetime] = None
        self._available: Optional[datetime] = None
        self._bid = self._ask = self._bid_size = self._ask_size = None
        self._session = None

    @property
    def open_bucket(self) -> bool:
        return self._start is not None

    def _progress(self) -> float:
        if self.kind == "volume":
            return self._volume
        if self.kind == "dollar":
            return self._dollar
        if self.kind == "tick":
            return float(self._trades)
        return (self._end - self._start).total_seconds() / 60.0 if self._start and self._end else 0.0

    def _emit(self, partial: bool = False) -> Bar:
        self._count += 1
        vwap = self._dollar / self._volume if self._volume > 0 else self._close
        minutes = max(1, int(round((self._end - self._start).total_seconds() / 60.0)))
        bar = Bar(self.symbol, self._start, self._open, self._high, self._low, self._close, self._volume, minutes,
                  self._bid, self._ask, quote_timestamp=None, observed_at=self._available if self._available > self._end else None,
                  bid_size=self._bid_size, ask_size=self._ask_size, end_time=self._end, dollar_volume=self._dollar,
                  trade_count=self._trades, vwap=vwap, bar_kind=self.kind,
                  bar_id=f"{self.symbol}-{self.kind}-{self._count:07d}{'-partial' if partial else ''}")
        self._reset()
        return bar

    def push(self, ev: MarketEvent) -> list[Bar]:
        """Add one event; returns the bars completed by it (0, 1 or 2 when a session boundary flushes)."""
        out: list[Bar] = []
        if ev.event_type == "QUOTE":
            self._bid, self._ask, self._bid_size, self._ask_size = ev.bid_price, ev.ask_price, ev.bid_size, ev.ask_size
            if self._start is not None:
                self._available = max(self._available or ev.available_at, ev.available_at)
            return out
        if ev.event_type == "BAR" and ev.bar is not None:
            b = ev.bar
            price_o, price_h, price_l, price_c, size = b.open, b.high, b.low, b.close, b.volume
            start, end = b.timestamp, b.close_time
            dollar = (b.dollar_volume if b.dollar_volume is not None else b.vwap * b.volume if b.vwap is not None else b.close * b.volume)
            trades = b.trade_count if b.trade_count is not None else 1
            quote = (b.bid, b.ask, b.bid_size, b.ask_size)
        else:
            price_o = price_h = price_l = price_c = float(ev.trade_price)
            size = float(ev.trade_size or 0.0)
            start = end = ev.exchange_timestamp
            dollar = price_c * size
            trades = 1
            quote = (self._bid, self._ask, self._bid_size, self._ask_size)
        session = self.calendar.session_date(start)
        if self._start is not None and session != self._session:
            out.append(self._emit(partial=True))          # never straddle a session boundary
        if self._start is None:
            self._start, self._session = start, session
            self._open, self._high, self._low = price_o, price_h, price_l
        self._high = max(self._high, price_h)
        self._low = min(self._low, price_l)
        self._close = price_c
        self._volume += size
        self._dollar += dollar
        self._trades += trades
        self._end = end
        self._available = max(self._available or ev.available_at, ev.available_at, end)
        self._bid, self._ask, self._bid_size, self._ask_size = quote
        if self._progress() >= self.target:
            out.append(self._emit())
        return out

    def flush(self) -> list[Bar]:
        return [self._emit(partial=True)] if self._start is not None else []


def build_event_bars(fine_bars: Iterable[Bar], kind: str, target: float, calendar: SessionCalendar | None = None,
                     flush_session_end: bool = True) -> list[Bar]:
    """Resample finer time bars into event bars.  ``target``: shares / notional / fine bars / minutes."""
    fine = list(fine_bars)
    if not fine:
        return []
    cal = calendar or SessionCalendar()
    builder = BarBuilder(kind, target, cal, fine[0].instrument)
    out: list[Bar] = []
    for ev in events_from_bars(fine):
        out.extend(builder.push(ev))
    if flush_session_end:
        out.extend(builder.flush())
    return out


class EventBarFeed:
    """Wraps a feed of finer bars (replay or live polling) and yields event bars as they complete."""

    def __init__(self, fine_feed: Iterable[Bar], kind: str, target: float, calendar: SessionCalendar | None = None,
                 symbol: str = ""):
        self.fine_feed = fine_feed
        self.builder = BarBuilder(kind, target, calendar, symbol)

    def __iter__(self) -> Iterator[Bar]:
        for b in self.fine_feed:
            if not self.builder.symbol:
                self.builder.symbol = b.instrument
            for ev in events_from_bars([b]):
                for bar in self.builder.push(ev):
                    yield bar

    def poll_new_bars(self, now: datetime | None = None) -> list[Bar]:  # pragma: no cover - live wrapper
        out: list[Bar] = []
        for b in self.fine_feed.poll_new_bars(now):
            if not self.builder.symbol:
                self.builder.symbol = b.instrument
            for ev in events_from_bars([b]):
                out.extend(self.builder.push(ev))
        return out


def empirical_bars_per_session(bars: Iterable[Bar], calendar: SessionCalendar | None = None) -> float:
    """Median number of bars per session date (event bars: the annualisation basis)."""
    cal = calendar or SessionCalendar()
    counts: dict = {}
    for b in bars:
        d = cal.session_date(b.timestamp)
        counts[d] = counts.get(d, 0) + 1
    if not counts:
        return 0.0
    vals = sorted(counts.values())
    return float(vals[len(vals) // 2])


__all__ = ["MarketEvent", "BarBuilder", "build_event_bars", "EventBarFeed", "events_from_bars", "empirical_bars_per_session",
           "BAR_KINDS"]
