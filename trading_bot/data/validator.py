"""DataValidator: data-quality circuit breakers (spec section 35).

Two failure classes:

* ``reject`` -- the bar itself is corrupt (non-finite/zero price, OHLC
  inconsistency, duplicate or backward timestamp, outside the session ...).
  It is never stored and no order is generated.
* ``halt``   -- the bar is structurally sound but the *sequence* is not
  (missing bar, extreme unvalidated jump).  The bar is stored as real data, but
  the bot enters DATA_HALTED and generates no new orders until
  ``halt_recovery_bars`` consecutive clean bars have arrived.

The validator never repairs or guesses a value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from ..types import Bar
from .calendar import SessionCalendar


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    storable: bool = True          # False => reject (do not append to the store)

    @property
    def halt_only(self) -> bool:
        return (not self.ok) and self.storable

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons), "storable": self.storable}


class DataValidator:
    def __init__(self, calendar: SessionCalendar, max_abs_log_jump: float = 0.10,
                 stale_feed_bars: int = 2, instrument: Optional[str] = None):
        self.calendar = calendar
        self.max_abs_log_jump = float(max_abs_log_jump)
        self.stale_feed_bars = int(stale_feed_bars)
        self.instrument = instrument
        self._last: Optional[Bar] = None

    @classmethod
    def from_config(cls, cfg, calendar: SessionCalendar | None = None) -> "DataValidator":
        cal = calendar or SessionCalendar.from_config(cfg)
        return cls(cal, cfg.data.max_abs_log_jump, cfg.data.stale_feed_bars, cfg.market.instrument)

    @property
    def last_bar(self) -> Optional[Bar]:
        return self._last

    def reset(self) -> None:
        self._last = None

    # ------------------------------------------------------------ checks
    def structural_reasons(self, bar: Bar) -> list[str]:
        reasons: list[str] = []
        if self.instrument is not None and bar.instrument != self.instrument:
            reasons.append(f"instrument mismatch: {bar.instrument} != {self.instrument}")
        vals = (bar.open, bar.high, bar.low, bar.close)
        if any(v is None or not math.isfinite(v) for v in vals):
            reasons.append("non-finite price")
        elif any(v <= 0 for v in vals):
            reasons.append("zero/negative price")
        else:
            if bar.high < bar.low:
                reasons.append("OHLC inconsistency: high < low")
            if not (bar.low - 1e-12 <= bar.open <= bar.high + 1e-12):
                reasons.append("OHLC inconsistency: open outside [low, high]")
            if not (bar.low - 1e-12 <= bar.close <= bar.high + 1e-12):
                reasons.append("OHLC inconsistency: close outside [low, high]")
        if bar.volume is None or not math.isfinite(bar.volume) or bar.volume < 0:
            reasons.append("invalid volume")
        if bar.bid is not None and bar.ask is not None:
            if not (math.isfinite(bar.bid) and math.isfinite(bar.ask)) or bar.bid <= 0 or bar.ask <= 0:
                reasons.append("invalid quote")
            elif bar.ask < bar.bid:
                reasons.append("crossed quote: ask < bid")
        if bar.timestamp.tzinfo is None:
            reasons.append("naive timestamp")
        elif not self.calendar.is_regular_session_bar(bar.timestamp):
            reasons.append("bar outside regular session")
        return reasons

    def sequence_reasons(self, bar: Bar, prev: Bar) -> tuple[list[str], list[str]]:
        """Return (reject_reasons, halt_reasons) for ``bar`` following ``prev``."""
        reject: list[str] = []
        halt: list[str] = []
        if bar.timestamp == prev.timestamp:
            reject.append("duplicate timestamp")
            return reject, halt
        if bar.timestamp < prev.timestamp:
            reject.append("timestamp moved backward")
            return reject, halt
        expected = self.calendar.expected_next_start(prev.timestamp)
        if expected is not None:
            if bar.timestamp != expected:
                if self.calendar.session_date(bar.timestamp) == self.calendar.session_date(prev.timestamp):
                    halt.append(f"missing bar: expected {expected.isoformat()}, got {bar.timestamp.isoformat()}")
                else:
                    halt.append(f"missing bar: session ended early after {prev.timestamp.isoformat()}")
        elif self.calendar.minutes_since_open(bar.timestamp) != 0:
            halt.append("missing bar: session did not start at the open")
        if prev.close > 0 and bar.close > 0:
            jump = abs(math.log(bar.close / prev.close))
            if jump > self.max_abs_log_jump:
                halt.append(f"extreme price jump: |log return| = {jump:.4f}")
        return reject, halt

    def check(self, bar: Bar, previous: Optional[Bar] = None) -> ValidationResult:
        """Validate ``bar`` against ``previous`` without mutating validator state."""
        prev = previous if previous is not None else self._last
        reject = self.structural_reasons(bar)
        halt: list[str] = []
        if prev is not None and not reject:
            seq_reject, halt = self.sequence_reasons(bar, prev)
            reject.extend(seq_reject)
        if reject:
            return ValidationResult(False, tuple(reject), storable=False)
        if halt:
            return ValidationResult(False, tuple(halt), storable=True)
        return ValidationResult(True, tuple(), storable=True)

    def validate(self, bar: Bar) -> ValidationResult:
        """Validate and, if the bar is storable, remember it as the latest bar."""
        result = self.check(bar)
        if result.storable:
            self._last = bar
        return result

    def accept(self, bar: Bar) -> None:
        self._last = bar

    def is_stale(self, now: datetime) -> bool:
        """Live-mode staleness: inside a session, more than ``stale_feed_bars`` completed bar
        intervals have elapsed since the later of the last bar close and the session open."""
        if self._last is None or not self.calendar.in_session(now):
            return False
        session_open = self.calendar.session_open_datetime(self.calendar.session_date(now))
        reference = max(self._last.close_time, session_open)
        limit = timedelta(minutes=self.calendar.bar_minutes * (self.stale_feed_bars + 1))
        return now - reference > limit


__all__ = ["DataValidator", "ValidationResult"]
