"""Market data feeds.

* ``ReplayFeed``    -- replays a stored bar history (backtests, tests).
* ``AlpacaBarFeed`` -- downloads historical 30-minute bars from Alpaca Market
  Data and polls for newly *completed* bars in paper-trading mode.  alpaca-py is
  imported lazily so the rest of the bot has no hard dependency on it.

Both feeds emit :class:`trading_bot.types.Bar` objects restricted to regular
session bars.  A bar is only ever emitted after it has closed (spec section 3);
the still-forming bar Alpaca returns for the current interval is dropped.  No
feed ever synthesizes a bar.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Iterator, Optional, Protocol

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
        from .store import read_bars_csv

        return cls(read_bars_csv(path, instrument, bar_minutes), calendar)


def _env(name: str, *alts: str) -> Optional[str]:
    for n in (name, *alts):
        v = os.environ.get(n)
        if v:
            return v
    return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AlpacaBarFeed:
    """Alpaca Market Data adapter (historical download + polling for completed bars).

    ``clock`` is injectable for tests.  ``completion_grace_seconds`` is the delay
    after a bar's close before it is considered final (Alpaca finalises the
    aggregate shortly after the interval ends).
    """

    def __init__(self, instrument: str, calendar: SessionCalendar, api_key: str | None = None,
                 secret_key: str | None = None, feed: str = "iex", bar_minutes: int = 30,
                 poll_seconds: int = 15, data_client=None, clock: Callable[[], datetime] = _utcnow,
                 completion_grace_seconds: int = 20, adjustment: str = "split"):
        self.instrument = instrument
        self.calendar = calendar
        self.bar_minutes = bar_minutes
        self.feed = feed
        self.poll_seconds = poll_seconds
        self.clock = clock
        self.completion_grace = timedelta(seconds=completion_grace_seconds)
        self.adjustment = adjustment
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

    @property
    def last_timestamp(self) -> Optional[datetime]:
        return self._last_ts

    def seed_last_timestamp(self, ts: datetime) -> None:
        self._last_ts = ts

    def _timeframe(self):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        return TimeFrame(self.bar_minutes, TimeFrameUnit.Minute)

    def _to_bars(self, raw_bars, horizon: datetime) -> list[Bar]:
        """Convert Alpaca bars; keep regular-session bars whose close is at or before ``horizon``."""
        out: list[Bar] = []
        for rb in raw_bars:
            ts = rb.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts = ts.astimezone(timezone.utc)
            if not self.calendar.is_regular_session_bar(ts):
                continue
            bar = Bar(self.instrument, ts, float(rb.open), float(rb.high), float(rb.low), float(rb.close),
                      float(rb.volume), self.bar_minutes)
            if bar.close_time > horizon:
                continue
            out.append(bar)
        out.sort(key=lambda b: b.timestamp)
        return out

    def fetch_history(self, start: datetime, end: datetime | None = None, now: datetime | None = None) -> list[Bar]:
        """Completed regular-session bars in [start, end].  Never advances the polling cursor."""
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest

        now = now or self.clock()
        horizon = now - self.completion_grace
        if end is not None:
            horizon = min(horizon, end)
        adjustment = {"split": Adjustment.SPLIT, "all": Adjustment.ALL, "raw": Adjustment.RAW,
                      "dividend": Adjustment.DIVIDEND}[self.adjustment]
        req = StockBarsRequest(symbol_or_symbols=self.instrument, timeframe=self._timeframe(), start=start, end=end,
                               adjustment=adjustment, feed=DataFeed(self.feed))
        resp = self.client.get_stock_bars(req)
        raw = resp.data.get(self.instrument, []) if hasattr(resp, "data") else resp[self.instrument]
        return self._to_bars(raw, horizon)

    def latest_quote(self) -> tuple[Optional[float], Optional[float], Optional[datetime]]:
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import StockLatestQuoteRequest

            q = self.client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=self.instrument,
                                                                           feed=DataFeed(self.feed)))
            quote = q[self.instrument]
            bid, ask = float(quote.bid_price), float(quote.ask_price)
            qts = getattr(quote, "timestamp", None)
            if qts is not None and qts.tzinfo is None:
                qts = qts.replace(tzinfo=timezone.utc)
            if bid > 0 and ask > 0 and ask >= bid:
                return bid, ask, qts
        except Exception:  # pragma: no cover - network dependent
            pass
        return None, None, None

    def poll_new_bars(self, now: datetime | None = None) -> list[Bar]:
        """Return completed regular-session bars newer than the polling cursor and advance it."""
        now = now or self.clock()
        prev = self._last_ts
        start = (prev + timedelta(minutes=self.bar_minutes)) if prev is not None else now - timedelta(days=7)
        bars = self.fetch_history(start, None, now)
        completed = [b for b in bars if prev is None or b.timestamp > prev]
        if not completed:
            return []
        self._last_ts = completed[-1].timestamp
        bid, ask, qts = self.latest_quote()
        if bid is not None:
            last = completed[-1]
            completed[-1] = Bar(last.instrument, last.timestamp, last.open, last.high, last.low, last.close,
                                last.volume, last.bar_minutes, bid, ask, quote_timestamp=qts or now)
        return completed

    def __iter__(self) -> Iterator[Bar]:  # pragma: no cover - long-running live loop
        while True:
            for b in self.poll_new_bars():
                yield b
            time.sleep(self.poll_seconds)


__all__ = ["MarketDataFeed", "ReplayFeed", "AlpacaBarFeed"]
