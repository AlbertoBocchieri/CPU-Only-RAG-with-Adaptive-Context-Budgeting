from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..utils import clamp


@dataclass(slots=True)
class LatencyCalibrator:
    prefill_target_ms: float = 18000.0
    cap_min_tokens: int = 512
    cap_max_tokens: int = 1536
    bootstrap_cap_tokens: int = 1024
    fixed_cap_tokens: int | None = None
    warmup_queries: int = 8
    ewma_alpha: float = 0.2
    query_count: int = 0
    ewma_prefill_ms_per_token: float | None = None
    ewma_decode_ms_per_token: float | None = None
    ewma_embed_ms_per_doc: float | None = None
    rss_mb_peak: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)

    def current_cap_tokens(self) -> int:
        if self.fixed_cap_tokens is not None:
            return int(round(clamp(float(self.fixed_cap_tokens), float(self.cap_min_tokens), float(self.cap_max_tokens))))
        if self.query_count < self.warmup_queries or self.ewma_prefill_ms_per_token is None:
            return int(self.bootstrap_cap_tokens)
        estimate = float(self.prefill_target_ms) / max(1e-6, float(self.ewma_prefill_ms_per_token))
        return int(round(clamp(estimate, float(self.cap_min_tokens), float(self.cap_max_tokens))))

    def current_cap_source(self) -> str:
        if self.fixed_cap_tokens is not None:
            return "fixed"
        if self.query_count < self.warmup_queries or self.ewma_prefill_ms_per_token is None:
            return "bootstrap"
        return "ewma_prefill"

    def update_prefill(self, *, context_tokens: int, prefill_ms: float) -> None:
        if int(context_tokens) <= 0 or float(prefill_ms) <= 0.0:
            self.query_count += 1
            return
        observed = float(prefill_ms) / max(1.0, float(context_tokens))
        if self.ewma_prefill_ms_per_token is None:
            self.ewma_prefill_ms_per_token = float(observed)
        else:
            alpha = clamp(float(self.ewma_alpha), 0.0, 1.0)
            prev = float(self.ewma_prefill_ms_per_token)
            self.ewma_prefill_ms_per_token = float((alpha * observed) + ((1.0 - alpha) * prev))
        self.trace.append({
            "kind": "prefill",
            "query_index": int(self.query_count),
            "observed_ms_per_token": float(observed),
            "ewma_prefill_ms_per_token": float(self.ewma_prefill_ms_per_token),
        })
        self.query_count += 1

    def update_decode(self, *, output_tokens: int, decode_ms: float) -> None:
        if int(output_tokens) <= 0 or float(decode_ms) <= 0.0:
            return
        observed = float(decode_ms) / max(1.0, float(output_tokens))
        if self.ewma_decode_ms_per_token is None:
            self.ewma_decode_ms_per_token = float(observed)
        else:
            alpha = clamp(float(self.ewma_alpha), 0.0, 1.0)
            prev = float(self.ewma_decode_ms_per_token)
            self.ewma_decode_ms_per_token = float((alpha * observed) + ((1.0 - alpha) * prev))

    def update_embedding(self, *, num_docs: int, embed_ms: float) -> None:
        if int(num_docs) <= 0 or float(embed_ms) <= 0.0:
            return
        observed = float(embed_ms) / max(1.0, float(num_docs))
        if self.ewma_embed_ms_per_doc is None:
            self.ewma_embed_ms_per_doc = float(observed)
        else:
            alpha = clamp(float(self.ewma_alpha), 0.0, 1.0)
            prev = float(self.ewma_embed_ms_per_doc)
            self.ewma_embed_ms_per_doc = float((alpha * observed) + ((1.0 - alpha) * prev))

    def update_rss(self, rss_mb: float) -> None:
        self.rss_mb_peak = max(float(self.rss_mb_peak), float(rss_mb))

    def snapshot(self) -> dict[str, Any]:
        return {
            "prefill_target_ms": float(self.prefill_target_ms),
            "cap_min_tokens": int(self.cap_min_tokens),
            "cap_max_tokens": int(self.cap_max_tokens),
            "bootstrap_cap_tokens": int(self.bootstrap_cap_tokens),
            "fixed_cap_tokens": (int(self.fixed_cap_tokens) if self.fixed_cap_tokens is not None else None),
            "warmup_queries": int(self.warmup_queries),
            "query_count": int(self.query_count),
            "ewma_prefill_ms_per_token": self.ewma_prefill_ms_per_token,
            "ewma_decode_ms_per_token": self.ewma_decode_ms_per_token,
            "ewma_embed_ms_per_doc": self.ewma_embed_ms_per_doc,
            "rss_mb_peak": float(self.rss_mb_peak),
            "budget_cap_tokens": int(self.current_cap_tokens()),
            "budget_cap_source": self.current_cap_source(),
        }
