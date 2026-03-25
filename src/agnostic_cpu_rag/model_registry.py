from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelSpec:
    role: str
    name: str
    backend: str
    params: dict[str, Any]


class ModelRegistry:
    def __init__(self, specs: dict[str, dict[str, ModelSpec]]):
        self.specs = specs

    @classmethod
    def load(cls, path: str) -> "ModelRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        models = raw.get("models", {})
        specs: dict[str, dict[str, ModelSpec]] = {}
        for role, entries in models.items():
            specs[str(role)] = {}
            for name, params in (entries or {}).items():
                p = dict(params or {})
                specs[str(role)][str(name)] = ModelSpec(
                    role=str(role),
                    name=str(name),
                    backend=str(p.pop("backend", "")),
                    params=p,
                )
        return cls(specs)

    def get(self, role: str, name: str) -> ModelSpec:
        try:
            return self.specs[str(role)][str(name)]
        except KeyError as exc:
            raise KeyError(f"Unknown model registry entry role={role} name={name}") from exc
