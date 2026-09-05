"""Immutable strategy configuration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("strategy.yaml")


class FrozenConfig(Mapping[str, Any]):
    """Read-only, attribute-accessible view over a nested mapping.

    Nested mappings are wrapped recursively; lists are converted to tuples so
    that no strategy constant can be mutated after loading.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]):
        frozen: dict[str, Any] = {}
        for key, value in data.items():
            frozen[str(key)] = _freeze(value)
        object.__setattr__(self, "_data", MappingProxyType(frozen))

    # Mapping protocol -----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("configuration is immutable")

    def __repr__(self) -> str:
        return f"FrozenConfig({dict(self._data)!r})"

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "FrozenConfig":
        """Return a new configuration with ``overrides`` deep-merged in."""
        merged = _deep_merge(self.to_dict(), _thaw(overrides))
        return FrozenConfig(merged)

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _freeze(value: Any) -> Any:
    if isinstance(value, FrozenConfig):
        return value
    if isinstance(value, Mapping):
        return FrozenConfig(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, FrozenConfig):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    if isinstance(value, list):
        return [_thaw(v) for v in value]
    return copy.deepcopy(value)


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = _thaw(value)
    return result


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> FrozenConfig:
    """Load the YAML strategy configuration into an immutable object."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = FrozenConfig(raw)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg


__all__ = ["FrozenConfig", "load_config", "DEFAULT_CONFIG_PATH"]
