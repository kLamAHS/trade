"""Explicit trading-session representation (spec section 2).

The calendar knows the exchange time zone and the regular session boundaries.
It never synthesizes bars; it only classifies timestamps and measures the
position of a bar inside its session (used by the time features, section 17).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def _parse_hhmm(text: str) -> time:
    hh, mm = text.split(":")
    return time(int(hh), int(mm))


@dataclass(frozen=True)
class SessionCalendar:
    timezone: str = "America/New_York"
    session_open: time = time(9, 30)
    session_close: time = time(16, 0)
    bar_minutes: int = 30

    @classmethod
    def from_config(cls, cfg) -> "SessionCalendar":
        m = cfg.market
        return cls(m.timezone, _parse_hhmm(m.session_open), _parse_hhmm(m.session_close), int(m.bar_minutes))

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def session_minutes(self) -> int:
        o = self.session_open.hour * 60 + self.session_open.minute
        c = self.session_close.hour * 60 + self.session_close.minute
        return c - o

    @property
    def bars_per_session(self) -> int:
        return self.session_minutes // self.bar_minutes

    def local(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(self.tz)

    def session_date(self, ts: datetime) -> date:
        return self.local(ts).date()

    def minutes_since_open(self, ts: datetime) -> int:
        loc = self.local(ts)
        return (loc.hour * 60 + loc.minute) - (self.session_open.hour * 60 + self.session_open.minute)

    def is_regular_session_bar(self, ts: datetime) -> bool:
        """True when a bar *starting* at ``ts`` lies inside the regular session."""
        loc = self.local(ts)
        if loc.weekday() >= 5:
            return False
        m = self.minutes_since_open(ts)
        return 0 <= m < self.session_minutes and (m % self.bar_minutes == 0)

    def session_open_datetime(self, d: date) -> datetime:
        return datetime.combine(d, self.session_open, tzinfo=self.tz)

    def session_close_datetime(self, d: date) -> datetime:
        return datetime.combine(d, self.session_close, tzinfo=self.tz)

    def expected_next_start(self, ts: datetime) -> datetime | None:
        """Next bar start inside the *same* session, or None if ``ts`` is the last bar."""
        nxt = ts + timedelta(minutes=self.bar_minutes)
        if self.session_date(nxt) != self.session_date(ts):
            return None
        if self.minutes_since_open(nxt) >= self.session_minutes:
            return None
        return nxt

    def is_session_boundary(self, prev_ts: datetime, ts: datetime) -> bool:
        return self.session_date(prev_ts) != self.session_date(ts)

    def regular_session_starts(self, d: date) -> list[datetime]:
        start = self.session_open_datetime(d)
        return [start + timedelta(minutes=i * self.bar_minutes) for i in range(self.bars_per_session)]


__all__ = ["SessionCalendar"]
