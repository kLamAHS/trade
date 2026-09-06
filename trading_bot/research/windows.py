"""Walk-forward window schedule and the locked final holdout (spec sections 6, 8).

Windows are expressed in *bar* indices of the full history.  For window k the model is
trained on bars ``[train_start_k, train_end_k)`` (everything selected inside that block by the
trainer's nested protocol), then traded on the unseen OOS block ``[train_end_k, oos_end_k)``.
The schedule stops before the locked holdout, which is only opened explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Window:
    index: int
    train_start: int
    train_end: int      # exclusive; first OOS bar
    oos_end: int        # exclusive

    @property
    def train_bars(self) -> int:
        return self.train_end - self.train_start

    @property
    def oos_bars(self) -> int:
        return self.oos_end - self.train_end

    def to_dict(self) -> dict:
        return {"index": self.index, "train_start": self.train_start, "train_end": self.train_end, "oos_end": self.oos_end}


@dataclass(frozen=True)
class WalkForwardSchedule:
    windows: tuple[Window, ...]
    holdout_start: Optional[int]     # first bar of the locked holdout (None = no holdout)
    n_bars: int
    train_bars: int
    oos_bars: int
    step_bars: int

    @property
    def development_end(self) -> int:
        return self.holdout_start if self.holdout_start is not None else self.n_bars

    def to_dict(self) -> dict:
        return {"windows": [w.to_dict() for w in self.windows], "holdout_start": self.holdout_start, "n_bars": self.n_bars,
                "train_bars": self.train_bars, "oos_bars": self.oos_bars, "step_bars": self.step_bars}


def build_schedule(n_bars: int, train_bars: int, oos_bars: int, step_bars: int | None = None,
                   holdout_fraction: float = 0.0, holdout_start: int | None = None, expanding: bool = False,
                   min_oos_bars: int | None = None, first_train_bars: int | None = None,
                   span: tuple[int, int] | None = None) -> WalkForwardSchedule:
    """Rolling (or expanding) walk-forward windows over the development span.

    ``holdout_start`` (bar index) or ``holdout_fraction`` (of all bars, from the end) defines the
    locked holdout.  The first window trains on ``first_train_bars`` (default ``train_bars``; the
    production bot starts at ``training.minimum_bars`` and grows to ``window_bars``), later windows
    on the trailing ``train_bars``.  ``span = (first_oos_bar, end)`` restricts the OOS blocks to a
    sub-range (used to walk the locked holdout once it is opened).  The last window is truncated
    to the span end; windows shorter than ``min_oos_bars`` (default: half of ``oos_bars``) are dropped.
    """
    step = step_bars or oos_bars
    if holdout_start is None and holdout_fraction > 0:
        holdout_start = int(np.floor(n_bars * (1.0 - holdout_fraction)))
    if span is not None:
        first_oos, dev_end = int(span[0]), int(span[1])
    else:
        first_oos, dev_end = int(first_train_bars or train_bars), (holdout_start if holdout_start is not None else n_bars)
    min_oos = min_oos_bars if min_oos_bars is not None else max(1, oos_bars // 2)
    windows: list[Window] = []
    train_end = first_oos
    k = 0
    while train_end < dev_end:
        oos_end = min(train_end + oos_bars, dev_end)
        if oos_end - train_end >= min_oos:
            start = 0 if expanding else max(0, train_end - train_bars)
            windows.append(Window(k, start, train_end, oos_end))
            k += 1
        train_end += step
    return WalkForwardSchedule(tuple(windows), holdout_start, n_bars, train_bars, oos_bars, step)


__all__ = ["Window", "WalkForwardSchedule", "build_schedule"]
