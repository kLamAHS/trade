"""Immutable value objects shared across the bot's modules.

Every object here is a frozen dataclass so that no downstream stage can mutate
the record produced by an upstream stage (spec sections 47-50).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional


class BotState(str, Enum):
    """The five operating states of the bot (spec section 44)."""

    INITIALIZING = "INITIALIZING"
    READY = "READY"
    POSITIONED = "POSITIONED"
    RISK_HALTED = "RISK_HALTED"
    DATA_HALTED = "DATA_HALTED"


@dataclass(frozen=True)
class Bar:
    """A completed OHLCV bar.  ``timestamp`` is the bar *start* (tz-aware, UTC)."""

    instrument: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_minutes: int = 30
    bid: Optional[float] = None
    ask: Optional[float] = None
    quote_timestamp: Optional[datetime] = None   # when bid/ask were observed (None => at the bar close)

    def __post_init__(self) -> None:
        # Normalise numeric fields to plain Python floats so that persistence,
        # checksums and JSON serialisation are exact and backend independent.
        for name in ("open", "high", "low", "close", "volume"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("bid", "ask"):
            v = getattr(self, name)
            object.__setattr__(self, name, None if v is None else float(v))
        if self.timestamp.tzinfo is None:
            raise ValueError("Bar.timestamp must be timezone-aware")

    @property
    def latest_source_time(self) -> datetime:
        """Newest information time carried by this bar (close, or a later quote)."""
        if self.quote_timestamp is not None and self.quote_timestamp > self.close_time:
            return self.quote_timestamp
        return self.close_time

    @property
    def close_time(self) -> datetime:
        return self.timestamp + timedelta(minutes=self.bar_minutes)

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return 0.5 * (self.bid + self.ask)

    @property
    def relative_spread(self) -> Optional[float]:
        mid = self.mid
        if mid is None or mid <= 0:
            return None
        return (self.ask - self.bid) / mid  # type: ignore[operator]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["quote_timestamp"] = self.quote_timestamp.isoformat() if self.quote_timestamp else None
        return d


@dataclass(frozen=True)
class FeatureVector:
    """Feature record for one instrument at one bar close (spec section 47).

    ``timestamp`` is the feature timestamp (bar close time).  ``latest_source_timestamp``
    is the close time of the newest bar that contributed to any value.  The engine
    refuses to emit a record where the source is newer than the feature (spec section 3).
    """

    instrument: str
    timestamp: datetime
    latest_source_timestamp: datetime
    bar_index: int
    fractional_d: float
    fractional_kernel_size: int
    values: Mapping[str, float]
    bar_close_time: Optional[datetime] = None   # close of the signal bar (== timestamp unless a later source exists)

    def __post_init__(self) -> None:
        if self.bar_close_time is None:
            object.__setattr__(self, "bar_close_time", self.timestamp)
        if self.latest_source_timestamp > self.timestamp:
            raise ValueError(
                "look-ahead violation: latest_source_timestamp "
                f"{self.latest_source_timestamp} > feature_timestamp {self.timestamp}"
            )
        # Feature values are read-only once the record is emitted.
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def get(self, name: str, default: float = math.nan) -> float:
        return self.values.get(name, default)

    def is_finite(self, names: Optional[list[str]] = None) -> bool:
        keys = names if names is not None else list(self.values.keys())
        return all(math.isfinite(self.values.get(k, math.nan)) for k in keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "timestamp": self.timestamp.isoformat(),
            "latest_source_timestamp": self.latest_source_timestamp.isoformat(),
            "bar_close_time": self.bar_close_time.isoformat() if self.bar_close_time else None,
            "bar_index": self.bar_index,
            "fractional_d": self.fractional_d,
            "fractional_kernel_size": self.fractional_kernel_size,
            **{k: (None if not math.isfinite(v) else float(v)) for k, v in self.values.items()},
        }


@dataclass(frozen=True)
class Prediction:
    """Model output for one bar (spec section 48)."""

    timestamp: datetime
    expected_normalized_return: float   # calibrated E_t
    expected_raw_return: float          # ER_t = E_t * sigma_50 * sqrt(H)
    probability_up: float               # P_t^+
    model_confidence: float             # |2P-1| when the models agree, else 0
    model_version: str
    regression_output: float = math.nan  # raw boosted output M_t
    combined_output: float = math.nan    # A_t = M_t * |D_t| (0 on disagreement)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass(frozen=True)
class CostEstimate:
    """Expected round-trip transaction cost decomposition (spec section 24)."""

    commission: float
    spread: float
    slippage: float

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.slippage

    def to_dict(self) -> dict[str, float]:
        return {"commission": self.commission, "spread": self.spread,
                "slippage": self.slippage, "total": self.total}


@dataclass(frozen=True)
class Signal:
    """Signal-engine output (spec section 49)."""

    timestamp: datetime
    direction: int
    expected_return: float
    estimated_cost: float
    expected_net_edge: float
    confidence: float
    target_exposure: float
    volatility_multiplier: float = 1.0
    reference_volatility: float = math.nan
    current_volatility: float = math.nan

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass(frozen=True)
class RiskDecision:
    """Risk-engine output (spec section 50).  Execution may only consume ``approved_exposure``."""

    proposed_exposure: float
    approved_exposure: float
    volatility_multiplier: float
    daily_loss_status: str
    drawdown_status: str
    max_holding_status: str
    reason: str
    stop_status: str = "OK"
    state: str = BotState.READY.value
    new_entry: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Order:
    """An exposure-change order queued for the next bar's open (spec section 42)."""

    instrument: str
    signal_timestamp: datetime
    side: str                  # "buy" | "sell"
    units: float               # absolute quantity
    target_exposure: float
    current_exposure: float
    estimated_price: float
    reason: str = "rebalance"
    order_id: str = ""
    new_entry: bool = False
    entry_sigma: float = math.nan

    @property
    def signed_units(self) -> float:
        return self.units if self.side == "buy" else -self.units

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["signal_timestamp"] = self.signal_timestamp.isoformat()
        return d


@dataclass(frozen=True)
class Fill:
    """An executed order (spec section 43)."""

    order_id: str
    instrument: str
    signal_timestamp: datetime
    fill_timestamp: datetime
    side: str
    units: float
    reference_price: float      # open of the execution bar (or Alpaca fill price basis)
    fill_price: float
    spread_cost: float          # currency units paid to the half spread
    slippage_cost: float        # currency units paid to slippage
    commission: float           # currency units
    source: str = "simulator"
    new_entry: bool = False
    entry_sigma: float = math.nan
    target_exposure: float = math.nan
    mirror: Optional[Mapping[str, Any]] = None   # broker (Alpaca) mirror status, annotation only
    price_source: str = "next_open"              # next_open | quote | broker: where the fill price was observed

    @property
    def signed_units(self) -> float:
        return self.units if self.side == "buy" else -self.units

    @property
    def notional(self) -> float:
        return self.units * self.fill_price

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["signal_timestamp"] = self.signal_timestamp.isoformat()
        d["fill_timestamp"] = self.fill_timestamp.isoformat()
        d["mirror"] = dict(self.mirror) if self.mirror is not None else None
        return d


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Point-in-time ledger state consumed by the risk engine."""

    timestamp: Optional[datetime]
    cash: float
    units: float
    mark_price: float
    equity: float
    exposure: float
    equity_peak: float
    drawdown: float
    session_start_equity: float
    daily_return: float
    realized_pnl: float
    unrealized_pnl: float
    entry_bar_index: Optional[int]
    entry_price: Optional[float]
    entry_sigma: Optional[float]
    holding_bars: int
    position_return: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp is not None else None
        return d


def dataclass_field_names(cls) -> list[str]:
    return [f.name for f in fields(cls)]


__all__ = [
    "BotState", "Bar", "FeatureVector", "Prediction", "CostEstimate", "Signal",
    "RiskDecision", "Order", "Fill", "PortfolioSnapshot", "dataclass_field_names", "field",
]
