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

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

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


# =====================================================================================================
# Hostile synthetic market (market-state spec sections 32-37)
# =====================================================================================================

REGIME_NAMES = ("calm", "trend_up", "trend_down", "mean_revert", "stress")


@dataclass
class SyntheticMarket:
    """Correlated synthetic instruments plus the generating *truth* (regimes, volatility, jumps, gaps,
    correlation, liquidity) that the engineering checks compare estimates against."""

    bars: dict[str, list[Bar]]
    truth: dict[str, np.ndarray]
    params: dict[str, Any]
    primary: str

    def __getitem__(self, symbol: str) -> list[Bar]:
        return self.bars[symbol]


def _student_t(rng: np.random.Generator, df: float, size) -> np.ndarray:
    """Unit-variance innovations: Student-t (fat tails) when ``df`` is finite and > 2, else normal."""
    if df is None or not np.isfinite(df) or df <= 2:
        return rng.standard_normal(size)
    return rng.standard_t(df, size) / math.sqrt(df / (df - 2.0))


def regime_sequence(n: int, rng: np.random.Generator, mean_bars: int, weights=None) -> np.ndarray:
    """Piecewise-constant regime index with geometric durations (mean ``mean_bars``)."""
    w = np.asarray(weights if weights is not None else (0.40, 0.15, 0.15, 0.15, 0.15), dtype=float)
    w = w / w.sum()
    out = np.zeros(n, dtype=int)
    i = 0
    prev = -1
    while i < n:
        k = int(rng.choice(len(REGIME_NAMES), p=w))
        if k == prev and len(REGIME_NAMES) > 1:
            k = int(rng.choice(len(REGIME_NAMES), p=w))
        length = int(rng.geometric(1.0 / max(1, mean_bars)))
        out[i: i + length] = k
        i += length
        prev = k
    return out


def generate_synthetic_market(n_bars: int, seed: int = 0, symbols=("SPY", "QQQ", "IWM"), primary: str = "SPY",
                              calendar: SessionCalendar | None = None, start: date | None = None,
                              base_vol: float = 0.0025, memory_d: float = 0.40, amplitude: float = 3.0,
                              stochastic_vol: float = 0.25, vol_persistence: float = 0.97,
                              regime_mean_bars: int = 260, regime_weights=None, jump_intensity: float = 0.002,
                              jump_size: float = 4.0, t_df: float = 4.0, gap_prob: float = 0.15, gap_size: float = 2.5,
                              correlation: float = 0.90, stress_correlation: float = 0.40,
                              lead_lag: dict | None = None, spread_bps: float = 1.0, stress_spread_mult: float = 3.0,
                              stress_range_mult: float = 2.0, quote_size: float = 500.0,
                              ofi_informativeness: float = 0.5, volume_burst_prob: float = 0.02,
                              trend_drift: float = 0.15, mean_revert_theta: float = 0.08,
                              start_price: float = 400.0, with_sizes: bool = True) -> SyntheticMarket:
    """Hostile multi-asset synthetic market.

    Per bar: a regime (calm / trend up / trend down / mean-reverting / stress) with geometric durations,
    mean-reverting stochastic log-volatility, fat-tailed innovations, Poisson jumps, overnight gaps,
    a common factor whose correlation collapses in stress, a liquidity process (spread and range
    multipliers, volume bursts) that co-moves with stress, bar-close NBBO quotes with displayed
    sizes whose imbalance carries ``ofi_informativeness`` about the next bar, and an optional
    lead/lag relation ``{"leader": "QQQ", "follower": "SPY", "beta": 0.3}`` (the follower's return
    loads on the leader's *previous* bar).  The primary keeps the long-memory component of
    :func:`generate_synthetic_bars`.  Everything is seeded and deterministic.
    """
    cal = calendar or SessionCalendar()
    rng = np.random.default_rng(seed)
    start = start or date(2022, 1, 3)
    symbols = tuple(symbols)
    if primary not in symbols:
        raise ValueError("primary must be one of symbols")
    n_sym = len(symbols)
    per_session = cal.bars_per_session
    season = 1.0 + 0.6 * np.abs(np.cos(np.pi * np.arange(per_session) / max(per_session - 1, 1)))
    # ---- calendar walk: timestamps and session slots
    stamps: list[datetime] = []
    slots: list[int] = []
    session_start_flags: list[bool] = []
    day = start
    while len(stamps) < n_bars:
        if day.weekday() >= 5 or cal.is_holiday(day):
            day += timedelta(days=1)
            continue
        for j, ts in enumerate(cal.regular_session_starts(day)):
            if len(stamps) >= n_bars:
                break
            stamps.append(ts.astimezone(timezone.utc))
            slots.append(j)
            session_start_flags.append(j == 0)
        day += timedelta(days=1)
    slots_a = np.asarray(slots)
    is_open = np.asarray(session_start_flags)
    # ---- regimes, volatility, liquidity
    regime = regime_sequence(n_bars, rng, regime_mean_bars, regime_weights)
    vol_mult = np.array([0.8, 1.0, 1.1, 0.9, 2.5])[regime]
    lv = np.zeros(n_bars)
    for t in range(1, n_bars):
        lv[t] = vol_persistence * lv[t - 1] + stochastic_vol * math.sqrt(1.0 - vol_persistence ** 2) * rng.standard_normal()
    vol = base_vol * np.exp(lv) * vol_mult * season[slots_a]
    spread_mult = np.where(regime == 4, stress_spread_mult, np.where(regime == 2, 1.5, 1.0))
    range_mult = np.where(regime == 4, stress_range_mult, 1.0)
    volume_mult = np.where(regime == 4, 2.0, 1.0) * np.where(rng.random(n_bars) < volume_burst_prob, 4.0, 1.0)
    rho = np.where(regime == 4, stress_correlation, correlation)
    jumps = rng.random(n_bars) < jump_intensity
    jump_sign = np.where(rng.random(n_bars) < 0.5, -1.0, 1.0)
    gaps = is_open & (rng.random(n_bars) < gap_prob)
    # ---- innovations
    common = _student_t(rng, t_df, n_bars)
    idio = _student_t(rng, t_df, (n_bars, n_sym))
    betas = {s: (1.0 if s == primary else float(rng.uniform(0.9, 1.3))) for s in symbols}
    u = fractional_noise(n_bars + 1, memory_d, rng) if amplitude > 0 else np.zeros(n_bars + 1)
    # ---- returns per symbol (log)
    drift = np.where(regime == 1, trend_drift, np.where(regime == 2, -trend_drift, 0.0)) * vol
    rets = np.zeros((n_bars, n_sym))
    walk = np.full(n_sym, math.log(start_price))
    levels = np.zeros((n_bars, n_sym))
    ma_window = 50
    leader_idx = follower_idx = None
    beta_ll = 0.0
    if lead_lag:
        leader_idx = symbols.index(lead_lag["leader"])
        follower_idx = symbols.index(lead_lag["follower"])
        beta_ll = float(lead_lag.get("beta", 0.3))
    for t in range(n_bars):
        f = common[t]
        for k in range(n_sym):
            eps = math.sqrt(rho[t]) * f + math.sqrt(max(0.0, 1.0 - rho[t])) * idio[t, k]
            r = vol[t] * betas[symbols[k]] * eps + drift[t]
            if regime[t] == 3 and t >= ma_window:
                r -= mean_revert_theta * (walk[k] - levels[t - ma_window: t, k].mean())
            if jumps[t]:
                r += jump_size * vol[t] * jump_sign[t]
            if gaps[t]:
                r += gap_size * vol[t] * common[t] * 0.7
            if follower_idx is not None and k == follower_idx and t > 0:
                r += beta_ll * rets[t - 1, leader_idx]
            rets[t, k] = r
            walk[k] += r
            levels[t, k] = walk[k]
    # long-memory component on the primary (as in generate_synthetic_bars)
    p_idx = symbols.index(primary)
    levels[:, p_idx] += amplitude * base_vol * u[1:]
    # ---- bars
    bars: dict[str, list[Bar]] = {s: [] for s in symbols}
    for k, sym in enumerate(symbols):
        prev_level = levels[0, k] - rets[0, k]
        for t in range(n_bars):
            v = vol[t]
            level = levels[t, k]
            open_p = float(math.exp(prev_level + 0.2 * v * rng.standard_normal()))
            close_p = float(math.exp(level))
            hi_extra = abs(rng.standard_normal()) * v * close_p * range_mult[t]
            lo_extra = abs(rng.standard_normal()) * v * close_p * range_mult[t]
            high = max(open_p, close_p) + hi_extra
            low = max(min(open_p, close_p) - lo_extra, 0.5 * close_p)
            volume = float(np.exp(12.0 + 0.5 * season[slots_a[t]] + 0.3 * rng.standard_normal()) * volume_mult[t])
            half = close_p * spread_bps * spread_mult[t] / 1e4 / 2.0
            bid, ask = float(close_p - half), float(close_p + half)
            bid_size = ask_size = None
            if with_sizes:
                next_r = rets[t + 1, k] if t + 1 < n_bars else 0.0
                signal = ofi_informativeness * next_r / (v + 1e-12) + 0.6 * rng.standard_normal()
                imb = float(np.tanh(signal))
                base_size = quote_size * float(np.exp(0.3 * rng.standard_normal())) / (1.0 if regime[t] != 4 else 2.0)
                bid_size = max(1.0, base_size * (1.0 + 0.8 * imb))
                ask_size = max(1.0, base_size * (1.0 - 0.8 * imb))
            bars[sym].append(Bar(sym, stamps[t], open_p, float(high), float(low), close_p, volume, cal.bar_minutes,
                                 bid, ask, bid_size=bid_size, ask_size=ask_size))
            prev_level = level
    truth = {"regime": regime, "regime_name": np.asarray([REGIME_NAMES[r] for r in regime], dtype=object), "vol": vol,
             "spread_mult": spread_mult, "range_mult": range_mult, "volume_mult": volume_mult, "correlation": rho,
             "jump": jumps, "gap": gaps, "returns": rets}
    params = {"n_bars": n_bars, "seed": seed, "symbols": list(symbols), "primary": primary, "base_vol": base_vol,
              "memory_d": memory_d, "amplitude": amplitude, "stochastic_vol": stochastic_vol, "vol_persistence": vol_persistence,
              "regime_mean_bars": regime_mean_bars, "jump_intensity": jump_intensity, "jump_size": jump_size, "t_df": t_df,
              "gap_prob": gap_prob, "gap_size": gap_size, "correlation": correlation, "stress_correlation": stress_correlation,
              "lead_lag": lead_lag, "spread_bps": spread_bps, "stress_spread_mult": stress_spread_mult,
              "ofi_informativeness": ofi_informativeness, "with_sizes": with_sizes}
    return SyntheticMarket(bars, truth, params, primary)


__all__ = ["generate_synthetic_bars", "fractional_noise", "generate_synthetic_market", "SyntheticMarket", "REGIME_NAMES",
           "regime_sequence"]
