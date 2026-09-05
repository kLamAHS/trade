"""Dashboard settings, including the Alpaca API credentials.

Settings are stored in a local JSON file (``settings.json`` next to the launcher by
default).  The file is git-ignored and written with owner-only permissions where
the platform supports it.  The secret key is never echoed back to the browser;
only a masked suffix is shown.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GuiSettings:
    api_key: str = ""
    secret_key: str = ""
    symbol: str = "SPY"
    mode: str = "backtest"            # backtest | paper
    data_source: str = "synthetic"    # synthetic | csv
    csv_path: str = ""
    synthetic_bars: int = 6000
    synthetic_seed: int = 0
    fast: bool = True
    mirror_orders: bool = True
    history_days: int = 1200
    artifacts_dir: str = "artifacts"
    initial_capital: float = 100000.0
    overrides: str = ""               # one "dotted.key=value" per line
    config_path: str = ""             # optional alternative strategy YAML

    PUBLIC_FIELDS = ("api_key", "symbol", "mode", "data_source", "csv_path", "synthetic_bars", "synthetic_seed",
                     "fast", "mirror_orders", "history_days", "artifacts_dir", "initial_capital",
                     "overrides", "config_path")

    # ------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: str | Path) -> "GuiSettings":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            with open(p, "r", encoding="utf-8") as fh:
                raw = json.load(fh) or {}
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        try:
            os.chmod(tmp, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass
        os.replace(tmp, p)

    # ------------------------------------------------------------- updates
    def update(self, data: dict[str, Any]) -> list[str]:
        """Apply a partial update from the browser; returns the names of changed fields.

        An empty ``secret_key`` leaves the stored secret untouched (the browser never has it).
        """
        changed: list[str] = []
        types = {f.name: f.type for f in fields(self)}
        for key, value in data.items():
            if key not in types or key == "PUBLIC_FIELDS" or key == "paper":
                continue          # 'paper' is not a setting: this application only ever uses the paper endpoint
            if key == "secret_key" and (value is None or value == ""):
                continue
            current = getattr(self, key)
            if isinstance(current, bool):
                value = bool(value) if not isinstance(value, str) else value.lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int) and not isinstance(current, bool):
                value = int(float(value))
            elif isinstance(current, float):
                value = float(value)
            else:
                value = "" if value is None else str(value).strip()
            if value != current:
                setattr(self, key, value)
                changed.append(key)
        return changed

    def public(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.PUBLIC_FIELDS}
        d["paper"] = True          # informational only; cannot be changed
        d["secret_key_set"] = bool(self.secret_key)
        d["secret_key_hint"] = ("•••• " + self.secret_key[-4:]) if len(self.secret_key) >= 8 else ("set" if self.secret_key else "")
        return d

    # ------------------------------------------------------------- helpers
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def apply_environment(self) -> None:
        """Expose the credentials to alpaca-py for this process only."""
        if self.api_key:
            os.environ["APCA_API_KEY_ID"] = self.api_key
        if self.secret_key:
            os.environ["APCA_API_SECRET_KEY"] = self.secret_key

    def override_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for line in (self.overrides or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            node = out
            parts = key.strip().split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            from ..main import parse_scalar

            node[parts[-1]] = parse_scalar(value)
        return out


__all__ = ["GuiSettings"]
