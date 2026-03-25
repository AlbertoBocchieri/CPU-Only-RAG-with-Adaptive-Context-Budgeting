from __future__ import annotations

from copy import deepcopy
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(base))
    for key, value in dict(override).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(dict(out[key]), value)
        else:
            out[key] = deepcopy(value)
    return out


def task_family_profile(cfg: dict[str, Any], task_family: str) -> dict[str, Any]:
    profiles = dict(dict(cfg).get("task_profiles", {}) or {})
    return dict(profiles.get(str(task_family), {}) or {})


def apply_task_family_profile(cfg: dict[str, Any], task_family: str) -> dict[str, Any]:
    base = deepcopy(dict(cfg))
    profiles = dict(base.pop("task_profiles", {}) or {})
    profile = dict(profiles.get(str(task_family), {}) or {})
    if not profile:
        return base
    return deep_merge(base, profile)


def resolved_utility_weights(cfg: dict[str, Any]) -> dict[str, float]:
    controller_cfg = dict(dict(cfg).get("context_controller", {}) or {})
    weights = dict(controller_cfg.get("utility_weights", {}) or {})
    return {str(key): float(value) for key, value in weights.items()}


def resolve_utility_weights_source(base_cfg: dict[str, Any], task_family: str) -> str:
    profile = task_family_profile(base_cfg, task_family)
    profile_controller = dict(profile.get("context_controller", {}) or {})
    if "utility_weights" in profile_controller:
        return "task_family_default"
    controller_cfg = dict(dict(base_cfg).get("context_controller", {}) or {})
    if "utility_weights" in controller_cfg:
        return "base_config"
    return "controller_builtin_default"
