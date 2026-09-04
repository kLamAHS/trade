"""Market data feeds.

* ``ReplayFeed``   -- replays a stored bar history (backtests, tests).
* ``AlpacaBarFeed`` -- downloads historical 30-minute bars from Alpaca Market
  Data and polls for newly completed bars in paper-trading mode.  alpaca-py is
  imported lazily so the rest of the bot has no hard dependency on it.

Both feeds emit :class:`trading_bot.types.Bar` objects restricted to regular
session bars.  No feed ever synthesizes a bar.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Optional, Protocol

from ..types import Bar
from .calendar import SessionCalendar
from .store import BarStore


class MarketDataFeed(Protocol):
    def __iter__(self) -> Iterator[Bar]: ...


class ReplayFeed:
    def __init__(self, bars: Iterable[Bar], calendar: SessionCalendar | None = None, regular_only: bool = True):
        self._bars = list(bars)
        self._cal = calendar or SessionCalendar()
        if regular_only:
            self._bars = [b for b in self._bars if self._cal.is_regular_session_bar(b.timestamp)]

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)

    def __len__(self) -> int:
        return len(self._bars)

    @classmethod
    def from_store(cls, store: BarStore, calendar: SessionCalendar | None = None) -> "ReplayFeed":
        return cls(store.bars, calendar)

    @classmethod
    def from_csv(cls, path: str, instrument: str, bar_minutes: int = 30,
                 calendar: SessionCalendar | None = None) -> "ReplayFeed":
        return cls.from_store(BarStore.load(path, instrument, bar_minutes), calendar)


def _env(name: str, *alts: str) -> Optional[str]:
    for n in (name, *alts):
        v = os.environ.get(n)
        if v:
            return v
    return None


class AlpacaBarFeed:
    """Alpaca Market Data adapter (historical + polling for new completed bars)."""

    def __init__(self, instrument: str, calendar: SessionCalendar, api_key: str | None = None,
                 secret_key: str | None = None, feed: str = "iex", bar_minutes: int = 30,
                 poll_seconds: int = 15, data_client=None):
        self.instrument = instrument
        self.calendar = calendar
        self.bar_minutes = bar_minutes
        self.feed = feed
        self.poll_seconds = poll_seconds
        self.api_key = api_key or _env("APCA_API_KEY_ID", "ALPACA_API_KEY")
        self.secret_key = secret_key or _env("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY")
        self._client = data_client
        self._last_ts: Optional[datetime] = None

    # ---------------------------------------------------------------- client
    @property
    def client(self):
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            if not self.api_key or not self.secret_key:
                raise RuntimeError("Alpaca credentials missing: set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
            self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self._client

    def _timeframe(self):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        return TimeFrame(self.bar_minutes, TimeFrameUnit.Minute)

    def _to_bars(self, raw_bars) -> list[Bar]:
        out: list[Bar] = []
        for rb in raw_bars:
            ts = rb.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts = ts.astimezone(timezone.utc)
            if not self.calendar.is_regular_session_bar(ts):
                continue
            out.append(Bar(self.instrument, ts, float(rb.open), float(rb.high), float(rb.low), float(rb.close),
                           float(rb.volume), self.bar_minutes))
        out.sort(key=lambda b: b.timestamp)
        return out

    def fetch_history(self, start: datetime, end: datetime | None = None) -> list[Bar]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import Adjustment, DataFeed

        req = StockBarsRequest(symbol_or_symbols=self.instrument, timeframe=self._timeframe(), start=start, end=end,
                               adjustment=Adjustment.ALL, feed=DataFeed(self.feed))
        resp = self.client.get_stock_bars(req)
        raw = resp.data.get(self.instrument, []) if hasattr(resp, "data") else resp[self.instrument]
        bars = self._to_bars(raw)
        if bars:
            self._last_ts = bars[-1].timestamp
        return bars

    def latest_quote(self) -> tuple[Optional[float], Optional[float]]:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.data.enums import DataFeed

            q = self.client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=self.instrument,
                                                                           feed=DataFeed(self.feed)))
            quote = q[self.instrument]
            bid, ask = float(quote.bid_price), float(quote.ask_price)
            if bid > 0 and ask > 0 and ask >= bid:
                return bid, ask
        except Exception:  # pragma: no cover - network dependent
            pass
        return None, None

    def poll_new_bars(self, now: datetime | None = None) -> list[Bar]:
        """Return completed regular-session bars newer than the last one seen."""
        now = now or datetime.now(timezone.utc)
        start = (self._last_ts + timedelta(minutes=self.bar_minutes)) if self._last_ts else now - timedelta(days=7)
        # Only bars that have *closed* are eligible; Alpaca may return the forming bar.
        bars = self.fetch_history(start, None)
        cutoff = now - timedelta(seconds=5)
        completed = [b for b in bars if b.close_time <= cutoff and (self._last_ts is None or b.timestamp > self._last_ts)]
        if completed:
            self._last_ts = completed[-1].timestamp
            bid, ask = self.latest_quote()
            if bid is not None:
                last = completed[-1]
                completed[-1] = Bar(last.instrument, last.timestamp, last.open, last.high, last.low, last.close,
                                    last.volume, last.bar_minutes, bid, ask)
        return completed

    def seed_last_timestamp(self, ts: datetime) -> None:
        self._last_ts = ts

    def __iter__(self) -> Iterator[Bar]:  # pragma: no cover - long-running live loop
        while True:
            for b in self.poll_new_bars():
                yield b
            time.sleep(self.poll_seconds)


__all__ = ["MarketDataFeed", "ReplayFeed", "AlpacaBarFeed"]
