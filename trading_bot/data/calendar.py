"""Explicit trading-session representation (spec section 2).

The calendar knows the exchange time zone, the regular session boundaries and
the early-close (half-day) sessions of the exchange.  It never synthesizes
bars; it only classifies timestamps and measures the position of a bar inside
its session (time features, section 17).

Early closes default to the NYSE rules (day after Thanksgiving, 3 July when it
is a Monday-Thursday, 24 December when it is a Monday-Thursday) and can be
extended or overridden from configuration (``market.early_closes``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def _parse_hhmm(text: str) -> time:
    hh, mm = str(text).split(":")
    return time(int(hh), int(mm))


def nyse_early_closes(year: int) -> set[date]:
    """Rule-based NYSE 13:00 early closes for ``year``."""
    out: set[date] = set()
    # Day after Thanksgiving (fourth Thursday of November).
    nov1 = date(year, 11, 1)
    first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
    out.add(first_thu + timedelta(weeks=3, days=1))
    july3 = date(year, 7, 3)
    if july3.weekday() <= 3:          # Mon-Thu (Independence Day observed on a weekday that follows)
        out.add(july3)
    dec24 = date(year, 12, 24)
    if dec24.weekday() <= 3:
        out.add(dec24)
    return out


@dataclass(frozen=True)
class SessionCalendar:
    timezone: str = "America/New_York"
    session_open: time = time(9, 30)
    session_close: time = time(16, 0)
    bar_minutes: int = 30
    early_close: time = time(13, 0)
    early_closes: frozenset[date] = field(default_factory=frozenset)   # explicit additions from config
    use_nyse_early_close_rules: bool = True

    @classmethod
    def from_config(cls, cfg) -> "SessionCalendar":
        m = cfg.market
        extra = frozenset(date.fromisoformat(str(d)) for d in (m.get("early_closes") or ()))
        return cls(m.timezone, _parse_hhmm(m.session_open), _parse_hhmm(m.session_close), int(m.bar_minutes),
                   _parse_hhmm(m.get("early_close_time", "13:00")), extra, bool(m.get("nyse_early_close_rules", True)))

    # ------------------------------------------------------------ basics
    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def local(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(self.tz)

    def session_date(self, ts: datetime) -> date:
        return self.local(ts).date()

    def is_early_close(self, d: date) -> bool:
        if d in self.early_closes:
            return True
        return self.use_nyse_early_close_rules and d in nyse_early_closes(d.year)

    def close_time_for(self, d: date) -> time:
        return self.early_close if self.is_early_close(d) else self.session_close

    def session_minutes_for(self, d: date) -> int:
        c = self.close_time_for(d)
        return (c.hour * 60 + c.minute) - (self.session_open.hour * 60 + self.session_open.minute)

    @property
    def session_minutes(self) -> int:
        """Regular (full-day) session length in minutes."""
        return (self.session_close.hour * 60 + self.session_close.minute) - (self.session_open.hour * 60 + self.session_open.minute)

    @property
    def bars_per_session(self) -> int:
        return self.session_minutes // self.bar_minutes

    def bars_for(self, d: date) -> int:
        return self.session_minutes_for(d) // self.bar_minutes

    def minutes_since_open(self, ts: datetime) -> int:
        loc = self.local(ts)
        return (loc.hour * 60 + loc.minute) - (self.session_open.hour * 60 + self.session_open.minute)

    def is_regular_session_bar(self, ts: datetime) -> bool:
        """True when a bar *starting* at ``ts`` lies inside that date's regular session."""
        loc = self.local(ts)
        if loc.weekday() >= 5:
            return False
        m = self.minutes_since_open(ts)
        return 0 <= m < self.session_minutes_for(loc.date()) and (m % self.bar_minutes == 0) and loc.second == 0

    def session_open_datetime(self, d: date) -> datetime:
        return datetime.combine(d, self.session_open, tzinfo=self.tz)

    def session_close_datetime(self, d: date) -> datetime:
        return datetime.combine(d, self.close_time_for(d), tzinfo=self.tz)

    def expected_next_start(self, ts: datetime) -> datetime | None:
        """Next bar start inside the *same* session, or None if ``ts`` is the last bar of its session."""
        nxt = ts + timedelta(minutes=self.bar_minutes)
        d = self.session_date(ts)
        if self.session_date(nxt) != d:
            return None
        if self.minutes_since_open(nxt) >= self.session_minutes_for(d):
            return None
        return nxt

    def is_session_boundary(self, prev_ts: datetime, ts: datetime) -> bool:
        return self.session_date(prev_ts) != self.session_date(ts)

    def regular_session_starts(self, d: date) -> list[datetime]:
        start = self.session_open_datetime(d)
        return [start + timedelta(minutes=i * self.bar_minutes) for i in range(self.bars_for(d))]

    def in_session(self, ts: datetime) -> bool:
        loc = self.local(ts)
        if loc.weekday() >= 5:
            return False
        return 0 <= self.minutes_since_open(ts) < self.session_minutes_for(loc.date())


__all__ = ["SessionCalendar", "nyse_early_closes"]
