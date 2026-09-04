"""Chronological walk-forward folds with purge and embargo (spec section 38).

Fold i (i = 1..k): train on the earliest (first_train + (i-1) step) fraction,
validate on the following ``step`` fraction.  ``purge`` rows are removed from
the end of the training block (their labels overlap the validation period) and
``embargo`` additional rows are skipped before validation starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray     # row indices
    validate: np.ndarray  # row indices

    @property
    def gap(self) -> int:
        return int(self.validate[0] - self.train[-1] - 1) if len(self.train) and len(self.validate) else 0


def walk_forward_folds(n_rows: int, folds: int = 5, first_train_fraction: float = 0.40,
                       validation_fraction: float = 0.10, purge: int = 5, embargo: int = 5) -> list[Fold]:
    out: list[Fold] = []
    for i in range(folds):
        boundary = int(np.floor(n_rows * (first_train_fraction + i * validation_fraction)))
        val_end = int(np.floor(n_rows * (first_train_fraction + (i + 1) * validation_fraction)))
        val_end = min(val_end, n_rows)
        train_end = boundary - purge
        val_start = boundary + embargo
        if train_end <= 0 or val_start >= val_end:
            raise ValueError(f"not enough rows ({n_rows}) for fold {i + 1} with purge={purge}, embargo={embargo}")
        out.append(Fold(i + 1, np.arange(0, train_end), np.arange(val_start, val_end)))
    return out


__all__ = ["Fold", "walk_forward_folds"]
