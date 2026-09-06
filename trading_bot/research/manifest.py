"""Deterministic run manifests and reproducibility checks (research spec sections 25-26).

A manifest fixes everything a run depends on: the data (hash), the configuration (two hashes:
the whole file, and the model-relevant sections), the code commit, the seed and the software
environment.  Two runs with the same manifest content must produce the same results hash; a
mismatch is reported as REPRODUCIBILITY FAILURE.  The run id is derived from the manifest so
that the same inputs always map to the same id prefix.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..training.trainer import environment_info, git_commit, software_version

MODEL_SECTIONS = ("seed", "market", "fractional", "features", "prediction", "models", "training", "signal", "risk", "execution")


def _digest(payload: Any, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:n]


def model_config_hash(cfg) -> str:
    d = cfg.to_dict()
    return _digest({k: d.get(k) for k in MODEL_SECTIONS})


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    kind: str                          # "real" | "synthetic"
    evidence_label: str
    instrument: str
    data: dict[str, Any]               # source, path, n_bars, span, data_hash
    config_hash: str
    model_config_hash: str
    code_commit: str
    software_version: str
    seed: int
    schedule: dict[str, Any]
    research: dict[str, Any]
    stages: list[str]
    environment: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""

    @classmethod
    def create(cls, cfg, store, data_info: dict[str, Any], schedule: dict[str, Any], stages: list[str],
               kind: str = "real", run_id: str | None = None) -> "RunManifest":
        env = environment_info()
        data = {"source": data_info.get("source", "unknown"), "path": data_info.get("path"), "n_bars": len(store),
                "span": [store[0].timestamp.isoformat(), store[-1].timestamp.isoformat()] if len(store) else None,
                "data_hash": store.checksum(), **{k: v for k, v in data_info.items() if k not in ("source", "path")}}
        research = cfg.get("research", {}) or {}
        research = research.to_dict() if hasattr(research, "to_dict") else dict(research)
        m = cls(run_id="", created_at=datetime.now(timezone.utc).isoformat(), kind=kind,
                evidence_label=("SYNTHETIC / ENGINEERING VALIDATION — NOT PERFORMANCE EVIDENCE" if kind == "synthetic"
                                else "REAL DATA / OUT-OF-SAMPLE EVIDENCE (subject to the acceptance gates)"),
                instrument=str(cfg.market.instrument), data=data, config_hash=cfg.digest(), model_config_hash=model_config_hash(cfg),
                code_commit=env["git_commit"], software_version=software_version(), seed=int(cfg.seed), schedule=schedule,
                research=research, stages=list(stages), environment=env, config=cfg.to_dict())
        m.manifest_hash = m.content_hash()
        m.run_id = run_id or f"{'syn' if kind == 'synthetic' else 'run'}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{m.manifest_hash[:6]}"
        return m

    def content_hash(self) -> str:
        """Hash of everything that determines the results (not the id or the creation time)."""
        d = asdict(self)
        for k in ("run_id", "created_at", "manifest_hash", "environment", "config"):
            d.pop(k, None)
        d["packages"] = self.environment.get("packages")
        d["python"] = self.environment.get("python")
        return _digest(d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)


def results_hash(oos_forecasts: dict[str, np.ndarray], equity: np.ndarray, model_hashes: list[str | None]) -> str:
    h = hashlib.sha256()
    for name in sorted(oos_forecasts):
        h.update(name.encode())
        h.update(np.ascontiguousarray(oos_forecasts[name], dtype=float).tobytes())
    h.update(np.ascontiguousarray(equity, dtype=float).tobytes())
    h.update("|".join(str(x) for x in model_hashes).encode())
    return h.hexdigest()[:16]


def compare_runs(summary_a: dict[str, Any], summary_b: dict[str, Any]) -> dict[str, Any]:
    """Two runs with identical manifest content must have identical results hashes."""
    ma, mb = summary_a.get("manifest", {}), summary_b.get("manifest", {})
    same_manifest = ma.get("manifest_hash") == mb.get("manifest_hash")
    same_results = summary_a.get("results_hash") == summary_b.get("results_hash")
    status = ("IDENTICAL" if same_manifest and same_results else "REPRODUCIBILITY FAILURE" if same_manifest
              else "DIFFERENT INPUTS")
    return {"status": status, "same_manifest": bool(same_manifest), "same_results": bool(same_results),
            "manifest_hash": [ma.get("manifest_hash"), mb.get("manifest_hash")],
            "results_hash": [summary_a.get("results_hash"), summary_b.get("results_hash")],
            "run_ids": [ma.get("run_id"), mb.get("run_id")]}


__all__ = ["RunManifest", "model_config_hash", "results_hash", "compare_runs", "MODEL_SECTIONS"]
