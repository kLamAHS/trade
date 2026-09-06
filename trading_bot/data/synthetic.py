"""Synthetic 30-minute bar generator with explicit session structure.

Log price = random walk + a stationary long-memory component:

    p_t = sum_{s<=t} vol_s eps_s  +  amplitude * vol * u_t,   u_t ~ ARFIMA(0, d, 0) (unit variance)

The random walk keeps prices I(1) (as in real markets) while the fractionally
integrated component ``u_t`` carries slowly decaying memory that a fractional
transform of order ~d can expose.  ``amplitude = 0`` gives a pure random walk
(no exploitable structure).  Intraday volatility is U-shaped; volume is a proxy.
Used for tests and the ``--synthetic`` demo; never for production decisions.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np

from ..types import Bar
from .calendar import SessionCalendar


def fractional_noise(n: int, d: float, rng: np.random.Generator, burn: int = 800, max_lag: int = 1000) -> np.ndarray:
    """Unit-variance ARFIMA(0, d, 0): (1-L)^{-d} eps_t via the GL kernel of order -d."""
    eps = rng.standard_normal(n + burn)
    if abs(d) < 1e-12:
        return eps[burn:]
    k = np.arange(1, max_lag)
    w = np.concatenate(([1.0], np.cumprod((d + k - 1.0) / k)))
    full = np.convolve(eps, w, mode="full")[: n + burn]
    u = full[burn:]
    return u / (np.std(u) + 1e-12)


def generate_synthetic_bars(n_bars: int, seed: int = 0, instrument: str = "SYN", start: date | None = None,
                            calendar: SessionCalendar | None = None, memory_d: float = 0.40,
                            amplitude: float = 3.0, base_vol: float = 0.0025, start_price: float = 400.0,
                            with_quotes: bool = True, spread_bps: float = 1.0, drift: float = 0.0,
                            autocorrelation: float = 0.0, jump_intensity: float = 0.0, jump_size: float = 0.0,
                            regime_bars: int = 0, regime_vol_ratio: float = 2.5, vol_clustering: float = 0.0) -> list[Bar]:
    """Optional market-structure knobs (research section 19); all default to off so the base
    process is unchanged:

    * ``drift``: annualised log drift of the random walk,
    * ``autocorrelation``: AR(1) coefficient of the random-walk innovations,
    * ``jump_intensity`` / ``jump_size``: per-bar jump probability and jump size (in units of vol),
    * ``regime_bars`` / ``regime_vol_ratio``: alternate low/high volatility regimes with geometric
      durations of that mean length; the high regime has ``regime_vol_ratio`` times the volatility,
    * ``vol_clustering``: GARCH-like persistence in [0, 1) of squared innovations.
    """
    cal = calendar or SessionCalendar()
    rng = np.random.default_rng(seed)
    start = start or date(2022, 1, 3)
    per_session = cal.bars_per_session
    slot = np.arange(per_session)
    season = 1.0 + 0.6 * np.abs(np.cos(np.pi * slot / max(per_session - 1, 1)))
    u = fractional_noise(n_bars + 1, memory_d, rng) if amplitude > 0 else np.zeros(n_bars + 1)
    structured = bool(drift or autocorrelation or jump_intensity or regime_bars or vol_clustering)
    rng2 = np.random.default_rng(seed + 1_000_003) if structured else None     # extra draws never disturb the base stream
    bars_per_year = per_session * 252
    mu = drift / bars_per_year
    bars: list[Bar] = []
    day = start
    i = 0
    walk = np.log(start_price)
    prev_level = walk + amplitude * base_vol * u[0]
    prev_eps = 0.0
    regime_high = False
    regime_left = 0
    cluster = 1.0
    while i < n_bars:
        if day.weekday() >= 5:
            day += timedelta(days=1)
            continue
        for j, ts in enumerate(cal.regular_session_starts(day)):
            if i >= n_bars:
                break
            vol = base_vol * season[j]
            eps = rng.standard_normal()
            if structured:
                if regime_bars > 0:
                    if regime_left <= 0:
                        regime_high = not regime_high
                        regime_left = int(rng2.geometric(1.0 / max(1, regime_bars)))
                    regime_left -= 1
                    if regime_high:
                        vol *= regime_vol_ratio
                if vol_clustering > 0:
                    cluster = (1.0 - vol_clustering) + vol_clustering * (0.5 * cluster + 0.5 * prev_eps * prev_eps)
                    vol *= float(np.sqrt(max(cluster, 0.05)))
                eps = autocorrelation * prev_eps + float(np.sqrt(max(1.0 - autocorrelation ** 2, 1e-6))) * eps
                if jump_intensity > 0 and rng2.random() < jump_intensity:
                    eps += jump_size * (1.0 if rng2.random() < 0.5 else -1.0)
                prev_eps = eps
            walk += mu + vol * eps
            level = walk + amplitude * base_vol * u[i + 1]
            open_p = float(np.exp(prev_level + 0.2 * vol * rng.standard_normal()))
            close_p = float(np.exp(level))
            hi_extra = abs(rng.standard_normal()) * vol * close_p
            lo_extra = abs(rng.standard_normal()) * vol * close_p
            high = max(open_p, close_p) + hi_extra
            low = max(min(open_p, close_p) - lo_extra, 0.5 * close_p)
            volume = float(np.exp(12.0 + 0.5 * season[j] + 0.3 * rng.standard_normal()))
            bid = ask = None
            if with_quotes:
                half = close_p * spread_bps / 1e4 / 2.0
                bid, ask = float(close_p - half), float(close_p + half)
            bars.append(Bar(instrument, ts.astimezone(timezone.utc), open_p, float(high), float(low), close_p,
                            volume, cal.bar_minutes, bid, ask))
            prev_level = level
            i += 1
        day += timedelta(days=1)
    return bars


__all__ = ["generate_synthetic_bars", "fractional_noise"]
