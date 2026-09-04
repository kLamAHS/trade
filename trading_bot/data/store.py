"""BarStore / HistoricalStore: append-only storage of validated bars.

Keeps parallel numpy arrays for fast feature computation and can persist to /
load from CSV.  The store never modifies a bar after it is appended.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from ..types import Bar


class BarStore:
    COLUMNS = ("timestamp", "open", "high", "low", "close", "volume", "bid", "ask")

    def __init__(self, instrument: str, bar_minutes: int = 30, bars: Iterable[Bar] | None = None):
        self.instrument = instrument
        self.bar_minutes = int(bar_minutes)
        self._bars: list[Bar] = []
        self._ts: list[datetime] = []
        self._open: list[float] = []
        self._high: list[float] = []
        self._low: list[float] = []
        self._close: list[float] = []
        self._volume: list[float] = []
        self._bid: list[float] = []
        self._ask: list[float] = []
        if bars is not None:
            for b in bars:
                self.append(b)

    # ------------------------------------------------------------------ basic
    def __len__(self) -> int:
        return len(self._bars)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)

    def __getitem__(self, idx):
        return self._bars[idx]

    @property
    def bars(self) -> Sequence[Bar]:
        return tuple(self._bars)

    def last(self, n: int | None = None):
        if n is None:
            return self._bars[-1] if self._bars else None
        return self.slice(max(0, len(self._bars) - n), len(self._bars))

    def append(self, bar: Bar) -> None:
        if self._ts and bar.timestamp <= self._ts[-1]:
            raise ValueError(f"bar timestamp {bar.timestamp} not after last {self._ts[-1]}")
        self._bars.append(bar)
        self._ts.append(bar.timestamp)
        self._open.append(bar.open)
        self._high.append(bar.high)
        self._low.append(bar.low)
        self._close.append(bar.close)
        self._volume.append(bar.volume)
        self._bid.append(math.nan if bar.bid is None else bar.bid)
        self._ask.append(math.nan if bar.ask is None else bar.ask)

    def extend(self, bars: Iterable[Bar]) -> None:
        for b in bars:
            self.append(b)

    def slice(self, start: int, stop: int) -> "BarStore":
        return BarStore(self.instrument, self.bar_minutes, self._bars[start:stop])

    # ----------------------------------------------------------------- arrays
    def arrays(self, start: int = 0, stop: int | None = None) -> dict[str, np.ndarray]:
        stop = len(self._bars) if stop is None else stop
        return {
            "open": np.asarray(self._open[start:stop], dtype=float),
            "high": np.asarray(self._high[start:stop], dtype=float),
            "low": np.asarray(self._low[start:stop], dtype=float),
            "close": np.asarray(self._close[start:stop], dtype=float),
            "volume": np.asarray(self._volume[start:stop], dtype=float),
            "bid": np.asarray(self._bid[start:stop], dtype=float),
            "ask": np.asarray(self._ask[start:stop], dtype=float),
            "timestamp": np.asarray(self._ts[start:stop], dtype=object),
        }

    def timestamps(self) -> list[datetime]:
        return list(self._ts)

    def log_close(self) -> np.ndarray:
        return np.log(np.asarray(self._close, dtype=float))

    def checksum(self, start: int = 0, stop: int | None = None) -> str:
        """Deterministic checksum of the stored OHLCV data (spec section 56)."""
        stop = len(self._bars) if stop is None else stop
        h = hashlib.sha256()
        for b in self._bars[start:stop]:
            h.update(f"{b.timestamp.isoformat()}|{b.open!r}|{b.high!r}|{b.low!r}|{b.close!r}|{b.volume!r}".encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------ persistence
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": [t.isoformat() for t in self._ts],
            "open": self._open, "high": self._high, "low": self._low, "close": self._close,
            "volume": self._volume, "bid": self._bid, "ask": self._ask,
        })

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False, float_format="%.17g")

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, instrument: str, bar_minutes: int = 30) -> "BarStore":
        """Strict constructor: raises on non-increasing timestamps (use ``bars_from_frame`` +
        the DataValidator for untrusted files)."""
        return cls(instrument, bar_minutes, bars_from_frame(frame, instrument, bar_minutes))

    @classmethod
    def load(cls, path: str | Path, instrument: str, bar_minutes: int = 30) -> "BarStore":
        return cls.from_frame(pd.read_csv(path, float_precision="round_trip"), instrument, bar_minutes)


def _opt(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def bars_from_frame(frame: pd.DataFrame, instrument: str, bar_minutes: int = 30) -> list[Bar]:
    """Rows in file order, without ordering checks: duplicates / backward timestamps are left
    for the DataValidator to classify (spec section 35)."""
    bars: list[Bar] = []
    for row in frame.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ts_py = ts.to_pydatetime().astimezone(timezone.utc)
        bars.append(Bar(instrument=instrument, timestamp=ts_py, open=float(row.open), high=float(row.high),
                        low=float(row.low), close=float(row.close), volume=float(row.volume),
                        bar_minutes=bar_minutes, bid=_opt(getattr(row, "bid", None)),
                        ask=_opt(getattr(row, "ask", None))))
    return bars


def read_bars_csv(path: str | Path, instrument: str, bar_minutes: int = 30) -> list[Bar]:
    return bars_from_frame(pd.read_csv(path, float_precision="round_trip"), instrument, bar_minutes)


HistoricalStore = BarStore

__all__ = ["BarStore", "HistoricalStore", "bars_from_frame", "read_bars_csv"]
