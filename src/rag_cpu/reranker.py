from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import CrossEncoder
from transformers import logging as hf_logging

from .types import RetrievedItem

hf_logging.set_verbosity_error()

_RERANKER_CACHE: dict[str, CrossEncoder] = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        if model_name in _RERANKER_CACHE:
            self.model = _RERANKER_CACHE[model_name]
        else:
            self.model = CrossEncoder(model_name)
            _RERANKER_CACHE[model_name] = self.model

    def rerank_with_trace(
        self,
        query: str,
        candidates: list[RetrievedItem],
        top_k_out: int,
    ) -> tuple[list[RetrievedItem], dict[str, object]]:
        t0 = time.perf_counter()
        if not candidates:
            return [], {"t_rerank_total_ms": 0.0}
        pairs = [[query, c.text] for c in candidates]
        scores = self.model.predict(pairs)
        ranked = list(zip(candidates, scores, strict=False))
        ranked.sort(key=lambda x: float(x[1]), reverse=True)

        out: list[RetrievedItem] = []
        for item, score in ranked[:top_k_out]:
            out.append(
                RetrievedItem(
                    item_id=item.item_id,
                    text=item.text,
                    score=float(score),
                    doc_id=item.doc_id,
                    source="rerank",
                )
            )
        t_rerank_total_ms = (time.perf_counter() - t0) * 1000.0
        return out, {"t_rerank_total_ms": float(t_rerank_total_ms)}

    def rerank(self, query: str, candidates: list[RetrievedItem], top_k_out: int) -> list[RetrievedItem]:
        out, _ = self.rerank_with_trace(query, candidates, top_k_out)
        return out
