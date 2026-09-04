"""ModelRegistry: persistent, atomically-promoted model artifacts (spec sections 37, 56)."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib

from .combined import CombinedModel, ModelMetadata


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current: Optional[CombinedModel] = None
        self._history: list[str] = []

    @property
    def current(self) -> Optional[CombinedModel]:
        with self._lock:
            return self._current

    @property
    def current_version(self) -> str:
        m = self.current
        return m.version if m is not None else "none"

    @property
    def history(self) -> list[str]:
        return list(self._history)

    def save(self, model: CombinedModel) -> Path:
        if model.metadata is None:
            raise ValueError("model must carry metadata before being saved")
        d = self.root / model.metadata.model_id
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, d / "model.joblib")
        with open(d / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(model.metadata.to_dict(), fh, indent=2, default=str)
        return d

    def promote(self, model: CombinedModel) -> None:
        """Atomically replace the live model.  Live trading keeps using the previous model until this call."""
        path = self.save(model)
        with self._lock:
            self._current = model
            self._history.append(model.metadata.model_id)
        tmp = self.root / "current.json.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"model_id": model.metadata.model_id, "path": str(path),
                       "promoted_at": datetime.now(timezone.utc).isoformat()}, fh)
        os.replace(tmp, self.root / "current.json")

    def load(self, model_id: str) -> CombinedModel:
        return joblib.load(self.root / model_id / "model.joblib")

    def load_current(self) -> Optional[CombinedModel]:
        p = self.root / "current.json"
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as fh:
            info = json.load(fh)
        model = self.load(info["model_id"])
        with self._lock:
            self._current = model
        return model

    def list_models(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            meta = d / "metadata.json"
            if meta.exists():
                with open(meta, "r", encoding="utf-8") as fh:
                    out.append(json.load(fh))
        return out


__all__ = ["ModelRegistry"]
