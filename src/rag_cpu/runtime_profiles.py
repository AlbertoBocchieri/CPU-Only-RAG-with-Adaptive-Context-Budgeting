from __future__ import annotations

from typing import Any


_SP3_PROFILES: dict[str, tuple[int, int]] = {
    "P4": (4, 4),
    "P4B6": (4, 6),
    "P6B6": (6, 6),
}


def resolve_llm_runtime(cfg: dict[str, Any]) -> dict[str, Any]:
    llm = cfg.get("llm", {})
    runtime = cfg.get("llm_runtime", {})

    sp3_enabled = bool(runtime.get("sp3_enabled", False))
    profile_raw = str(runtime.get("sp3_profile", "BASELINE")).upper()
    profile = profile_raw if profile_raw in {"BASELINE", "P4", "P4B6", "P6B6"} else "BASELINE"

    base_threads_decode = int(runtime.get("threads_decode", llm.get("n_threads", 4)))
    base_threads_batch = int(runtime.get("threads_batch", base_threads_decode))

    if sp3_enabled and profile in _SP3_PROFILES:
        threads_decode, threads_batch = _SP3_PROFILES[profile]
    else:
        profile = "BASELINE"
        threads_decode, threads_batch = base_threads_decode, base_threads_batch

    batch_size = runtime.get("batch_size", None)
    ubatch_size = runtime.get("ubatch_size", None)

    effective_batch = int(batch_size) if batch_size is not None else int(llm.get("n_batch", 512))
    effective_ubatch = int(ubatch_size) if ubatch_size is not None else None

    return {
        "enabled": bool(sp3_enabled),
        "profile": profile,
        "threads_decode": int(threads_decode),
        "threads_batch": int(threads_batch),
        "batch_size": int(effective_batch),
        "ubatch_size": effective_ubatch,
    }

