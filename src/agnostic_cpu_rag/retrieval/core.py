from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rag_cpu.reranker import CrossEncoderReranker
from rag_cpu.retrievers import BM25Retriever, DenseRetriever, IndexItem, RetrievedItem

from ..model_registry import ModelRegistry
from ..records import DocumentRecord, RetrievedCandidate
from ..runtime.calibration import LatencyCalibrator
from .fusion import compute_dynamic_alpha, fuse_weighted_sum
from .hop2 import build_hop2_queries, conservative_novel_slot_merge, extract_bridge_terms, should_activate_hop2


@dataclass(slots=True)
class RetrievalOutput:
    candidates: list[RetrievedCandidate]
    trace: dict[str, Any]


class AgnosticRetriever:
    def __init__(
        self,
        *,
        documents: dict[str, DocumentRecord],
        config: dict[str, Any],
        model_registry: ModelRegistry,
    ):
        self.documents = documents
        self.config = dict(config or {})
        dense_enabled = bool(self.config.get("dense_enabled", True))
        bm25_enabled = bool(self.config.get("bm25_enabled", True))
        dense_model_name = str(self.config.get("dense_model_name", "e5_small"))
        reranker_enabled = bool(self.config.get("reranker_enabled", False))
        reranker_model_name = str(self.config.get("reranker_model_name", "msmarco_minilm"))

        items = {
            doc_id: IndexItem(item_id=doc_id, doc_id=doc_id, text=f"{doc.title}\n{doc.text}".strip())
            for doc_id, doc in documents.items()
            if doc.text.strip()
        }
        self.items = items
        self.bm25 = BM25Retriever(items) if bm25_enabled else None
        self.dense = None
        if dense_enabled:
            dense_spec = model_registry.get("dense", dense_model_name)
            self.dense = DenseRetriever(
                items,
                model_name=str(dense_spec.params["model_name"]),
                query_prefix=str(dense_spec.params.get("query_prefix", "")),
                passage_prefix=str(dense_spec.params.get("passage_prefix", "")),
                normalize=bool(self.config.get("normalize_scores", True)),
                persist_cache=bool(self.config.get("persist_embedding_cache", True)),
                cache_dir=str(self.config.get("embedding_cache_dir", "cache/embeddings")),
            )
        self.reranker = None
        if reranker_enabled:
            rerank_spec = model_registry.get("reranker", reranker_model_name)
            self.reranker = CrossEncoderReranker(str(rerank_spec.params["model_name"]))

    def _convert(self, hits: list[RetrievedItem], source: str) -> list[RetrievedCandidate]:
        out: list[RetrievedCandidate] = []
        for hit in hits:
            doc = self.documents.get(hit.doc_id)
            out.append(
                RetrievedCandidate(
                    item_id=str(hit.item_id),
                    doc_id=str(hit.doc_id),
                    text=str(hit.text),
                    score=float(hit.score),
                    source=source,
                    title=str(doc.title if doc else ""),
                    metadata={"source": source},
                )
            )
        return out

    def _search_channels(self, query: str) -> tuple[list[RetrievedItem], dict[str, Any], list[RetrievedItem], dict[str, Any]]:
        top_k_dense = int(self.config.get("top_k_dense", 20))
        top_k_bm25 = int(self.config.get("top_k_bm25", 20))
        dense_hits: list[RetrievedItem] = []
        dense_trace: dict[str, Any] = {"t_query_embed_ms": 0.0, "t_vector_search_ms": 0.0}
        bm25_hits: list[RetrievedItem] = []
        bm25_trace: dict[str, Any] = {"t_bm25_search_ms": 0.0}
        if self.dense is not None:
            dense_hits, dense_trace = self.dense.search_with_trace(query, top_k_dense)
        if self.bm25 is not None:
            bm25_hits, bm25_trace = self.bm25.search_with_trace(query, top_k_bm25)
        return dense_hits, dense_trace, bm25_hits, bm25_trace

    def _fuse(self, dense_hits: list[RetrievedItem], bm25_hits: list[RetrievedItem]) -> tuple[list[RetrievedCandidate], dict[str, Any]]:
        dynamic_alpha_enabled = bool(self.config.get("dynamic_alpha_enabled", False))
        fixed_alpha = float(self.config.get("fixed_alpha", 0.5))
        low_conf_threshold = float(self.config.get("low_conf_threshold", 0.2))
        top_k_final = int(self.config.get("top_k_final", 10))

        id_to_hit: dict[str, RetrievedItem] = {}
        for hit in dense_hits:
            id_to_hit[hit.item_id] = hit
        for hit in bm25_hits:
            id_to_hit.setdefault(hit.item_id, hit)

        if dense_hits and bm25_hits:
            if dynamic_alpha_enabled:
                alpha_q, alpha_trace = compute_dynamic_alpha(
                    dense_hits=dense_hits,
                    lexical_hits=bm25_hits,
                    low_conf_threshold=low_conf_threshold,
                )
            else:
                _, dynamic_trace = compute_dynamic_alpha(
                    dense_hits=dense_hits,
                    lexical_hits=bm25_hits,
                    low_conf_threshold=low_conf_threshold,
                )
                alpha_q = float(fixed_alpha)
                alpha_trace = {
                    "alpha_q": float(alpha_q),
                    "alpha_source": "fixed",
                    "conf_dense": float(dynamic_trace.get("conf_dense", 0.0)),
                    "conf_lexical": float(dynamic_trace.get("conf_lexical", 0.0)),
                    "agreement": float(dynamic_trace.get("agreement", 0.0)),
                    "dense_gap": float(dynamic_trace.get("dense_gap", 0.0)),
                    "lexical_gap": float(dynamic_trace.get("lexical_gap", 0.0)),
                    "dense_concentration": float(dynamic_trace.get("dense_concentration", 0.0)),
                    "lexical_concentration": float(dynamic_trace.get("lexical_concentration", 0.0)),
                    "low_conf_threshold": float(low_conf_threshold),
                }
            fused_scores = fuse_weighted_sum(dense_hits=dense_hits, lexical_hits=bm25_hits, alpha=alpha_q)
            ranked = sorted(fused_scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:top_k_final]
            out = self._convert(
                [
                    RetrievedItem(
                        item_id=item_id,
                        text=id_to_hit[item_id].text,
                        score=float(score),
                        doc_id=id_to_hit[item_id].doc_id,
                        source="hybrid",
                    )
                    for item_id, score in ranked
                ],
                source="hybrid",
            )
            return out, alpha_trace
        if dense_hits:
            return self._convert(dense_hits[:top_k_final], source="dense_only"), {
                "alpha_q": 1.0,
                "alpha_source": "dense_only",
                "conf_dense": 1.0,
                "conf_lexical": 0.0,
                "agreement": None,
            }
        return self._convert(bm25_hits[:top_k_final], source="bm25_only"), {
            "alpha_q": 0.0,
            "alpha_source": "bm25_only",
            "conf_dense": 0.0,
            "conf_lexical": 1.0,
            "agreement": None,
        }

    def search(self, *, query: str, calibrator: LatencyCalibrator | None = None) -> RetrievalOutput:
        t0 = time.perf_counter()
        dense_hits, dense_trace, bm25_hits, bm25_trace = self._search_channels(query)
        if calibrator is not None:
            calibrator.update_embedding(num_docs=max(1, len(self.documents)), embed_ms=float(dense_trace.get("t_query_embed_ms", 0.0)))
        base_candidates, alpha_trace = self._fuse(dense_hits, bm25_hits)

        hop2_cfg = dict(self.config.get("hop2", {}))
        hop2_trace: dict[str, Any] = {"enabled": bool(hop2_cfg.get("enabled", False)), "used": False}
        final_candidates = list(base_candidates)
        if bool(hop2_cfg.get("enabled", False)) and (self.dense is not None or self.bm25 is not None):
            retrieval_budget_ms = float(hop2_cfg.get("retrieval_budget_ms", 1500.0))
            estimated_extra_cost_ms = float(dense_trace.get("t_query_embed_ms", 0.0)) + float(dense_trace.get("t_vector_search_ms", 0.0)) + float(bm25_trace.get("t_bm25_search_ms", 0.0))
            activate = should_activate_hop2(
                fusion_trace=alpha_trace,
                retrieval_budget_ms=retrieval_budget_ms,
                estimated_extra_cost_ms=estimated_extra_cost_ms,
                activation_conf_threshold=float(hop2_cfg.get("activation_conf_threshold", 0.35)),
                agreement_threshold=float(hop2_cfg.get("agreement_threshold", 0.2)),
            )
            bridge_terms = extract_bridge_terms(query, base_candidates[: int(hop2_cfg.get("top_seed_hits", 3))], max_terms=int(hop2_cfg.get("max_bridge_terms", 2)))
            queries = build_hop2_queries(query, bridge_terms, max_queries=int(hop2_cfg.get("max_queries", 2))) if activate else []
            hop2_ranked: list[RetrievedCandidate] = []
            hop2_details: list[dict[str, Any]] = []
            for expanded_query in queries:
                dh2, dense_trace_h2, bh2, bm25_trace_h2 = self._search_channels(expanded_query)
                hop2_candidates, hop2_alpha = self._fuse(dh2, bh2)
                hop2_ranked.extend(hop2_candidates)
                hop2_details.append(
                    {
                        "expanded_query": expanded_query,
                        "alpha_trace": hop2_alpha,
                        "dense_trace": dense_trace_h2,
                        "bm25_trace": bm25_trace_h2,
                        "candidate_ids": [c.item_id for c in hop2_candidates],
                    }
                )
            if hop2_ranked:
                final_candidates, merge_trace = conservative_novel_slot_merge(
                    base_ranked=base_candidates,
                    hop2_ranked=hop2_ranked,
                    top_k_final=int(self.config.get("top_k_final", 10)),
                    reserved_novel_slots=int(hop2_cfg.get("reserved_novel_slots", 2)),
                    min_normalized_score=float(hop2_cfg.get("min_normalized_score", 0.35)),
                )
                hop2_trace = {
                    "enabled": True,
                    "used": True,
                    "bridge_terms": bridge_terms,
                    "queries": queries,
                    "details": hop2_details,
                    "merge": merge_trace,
                }
            else:
                hop2_trace = {
                    "enabled": True,
                    "used": False,
                    "bridge_terms": bridge_terms,
                    "queries": queries,
                }

        rerank_trace = {"enabled": False, "t_rerank_total_ms": 0.0}
        if self.reranker is not None and final_candidates:
            rerank_in = final_candidates[: int(self.config.get("reranker_top_k_in", len(final_candidates)))]
            reranked, raw_trace = self.reranker.rerank_with_trace(
                query,
                [
                    RetrievedItem(
                        item_id=c.item_id,
                        text=c.text,
                        score=float(c.score),
                        doc_id=c.doc_id,
                        source=c.source,
                    )
                    for c in rerank_in
                ],
                top_k_out=int(self.config.get("top_k_final", 10)),
            )
            final_candidates = self._convert(reranked, source="rerank")
            rerank_trace = {"enabled": True, **raw_trace}

        trace = {
            "dense": dense_trace,
            "bm25": bm25_trace,
            "fusion": alpha_trace,
            "hop2": hop2_trace,
            "rerank": rerank_trace,
            "final_item_ids": [c.item_id for c in final_candidates],
            "final_doc_ids": [c.doc_id for c in final_candidates],
            "t_retrieval_total_ms": float((time.perf_counter() - t0) * 1000.0),
        }
        return RetrievalOutput(candidates=final_candidates, trace=trace)
