from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .context_budgeting import (
    adaptive_pack_contexts,
    resolve_context_budgeting_config,
    resolve_margin_threshold,
)
from .generator import LlamaCppGenerator
from .reranker import CrossEncoderReranker
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, IndexItem
from .runtime_profiles import resolve_llm_runtime
from .types import RetrievedItem


@dataclass(slots=True)
class PipelineOutput:
    retrieved: list[RetrievedItem]
    answer: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retrieval_stages: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    token_stats: dict[str, float] = field(default_factory=dict)
    context_budgeting: dict[str, Any] = field(default_factory=dict)
    post_context: dict[str, Any] = field(default_factory=dict)
    sp3: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, cfg: dict, items: dict[str, IndexItem], enable_llm: bool):
        self.cfg = cfg
        retrieval_cfg = cfg["retrieval"]
        retriever_mode = str(retrieval_cfg.get("retriever_mode", "hybrid")).lower()
        bm25 = BM25Retriever(items) if retriever_mode in {"hybrid", "bm25_only"} else None
        dense = (
            DenseRetriever(
                items,
                model_name=retrieval_cfg["dense_model"],
                query_prefix=retrieval_cfg.get("dense_prefix_query", ""),
                passage_prefix=retrieval_cfg.get("dense_prefix_passage", ""),
                normalize=bool(retrieval_cfg.get("normalize_scores", True)),
                persist_cache=bool(retrieval_cfg.get("persist_embedding_cache", True)),
                cache_dir=str(retrieval_cfg.get("embedding_cache_dir", "cache/embeddings")),
            )
            if retriever_mode in {"hybrid", "dense_only"}
            else None
        )
        self.hybrid = HybridRetriever(
            bm25=bm25,
            dense=dense,
            alpha=float(retrieval_cfg.get("hybrid_alpha", 0.5)),
            fusion_method=str(retrieval_cfg.get("fusion_method", "RRF")),
            rrf_k=int(retrieval_cfg.get("rrf_k", 60)),
            normalize_scores=bool(retrieval_cfg.get("normalize_scores", True)),
        )

        rer_cfg = cfg["reranker"]
        self.reranker = (
            CrossEncoderReranker(rer_cfg["model_name"]) if bool(rer_cfg.get("enabled", False)) else None
        )

        self.enable_llm = enable_llm
        self.generator = None
        self.runtime_profile = resolve_llm_runtime(cfg)
        context_budget_cfg = cfg.get("context_budgeting", {})
        self.context_budgeting_resolved = resolve_context_budgeting_config(context_budget_cfg)
        self.context_budgeting_enabled = bool(self.context_budgeting_resolved.get("enabled", False))
        self.context_budgeting_strategy = str(self.context_budgeting_resolved.get("strategy", "v1"))
        self.context_budget_margin_threshold = 0.003
        self.context_budget_threshold_meta: dict[str, Any] = {
            "margin_threshold_source": "fallback",
            "margin_threshold_quantile": float(context_budget_cfg.get("margin_threshold_quantile", 0.9)),
            "margin_threshold_stage2_glob": str(context_budget_cfg.get("margin_threshold_stage2_glob", "")),
            "margin_threshold_retriever_mode": str(context_budget_cfg.get("margin_threshold_stage2_retriever_mode", "hybrid")),
            "margin_threshold_samples": 0,
        }
        self.sc_query_count = 0
        self.sc_ewma_prefill_ms_per_token: float | None = None
        if self.context_budgeting_enabled:
            if self.context_budgeting_strategy in {"incremental_sc", "agnostic_acb_sc"}:
                self.context_budget_margin_threshold = 0.0
                self.context_budget_threshold_meta = {
                    "margin_threshold_source": "query_local",
                    "margin_threshold_quantile": None,
                    "margin_threshold_stage2_glob": "",
                    "margin_threshold_retriever_mode": "",
                    "margin_threshold_samples": 0,
                }
            else:
                threshold, threshold_meta = resolve_margin_threshold(context_budget_cfg)
                self.context_budget_margin_threshold = float(threshold)
                self.context_budget_threshold_meta = dict(threshold_meta)
        if enable_llm:
            llm_cfg = cfg["llm"]
            self.generator = LlamaCppGenerator(
                gguf_path=llm_cfg["gguf_path"],
                n_ctx=int(llm_cfg["n_ctx"]),
                n_threads=int(self.runtime_profile["threads_decode"]),
                n_batch=int(self.runtime_profile["batch_size"]),
                n_threads_batch=int(self.runtime_profile["threads_batch"]),
                n_ubatch=self.runtime_profile.get("ubatch_size"),
                prefix_cache_enabled=bool(llm_cfg.get("prefix_cache_enabled", False)),
                prefix_cache_backend=str(llm_cfg.get("prefix_cache_backend", "ram")),
                prefix_cache_capacity_mb=int(llm_cfg.get("prefix_cache_capacity_mb", 256)),
                prefix_cache_dir=str(llm_cfg.get("prefix_cache_dir", "cache/llama_prompt_cache")),
            )

    @staticmethod
    def _pack_contexts(contexts: list[str], cfg: dict) -> tuple[list[str], float]:
        llm_cfg = cfg["llm"]
        enabled = bool(llm_cfg.get("context_packing", False))
        max_words = int(llm_cfg.get("context_pack_words", 80))
        if not enabled:
            return contexts, 0.0
        t0 = time.perf_counter()
        packed: list[str] = []
        for ctx in contexts:
            words = ctx.split()
            packed.append(" ".join(words[:max_words]))
        return packed, (time.perf_counter() - t0) * 1000.0

    def retrieve_with_trace(self, question: str, cfg: dict) -> tuple[list[RetrievedItem], dict[str, Any], dict[str, Any]]:
        ret_cfg = cfg["retrieval"]
        rer_cfg = cfg["reranker"]
        hits, trace = self.hybrid.search_with_trace(
            question,
            top_k_dense=int(ret_cfg["top_k_dense"]),
            top_k_bm25=int(ret_cfg["top_k_bm25"]),
            top_k_final=int(ret_cfg["top_k_final"]),
            retriever_mode=str(ret_cfg.get("retriever_mode", "hybrid")),
            fusion_method=str(ret_cfg.get("fusion_method", "RRF")),
            rrf_k=int(ret_cfg.get("rrf_k", 60)),
            weighted_alpha=float(ret_cfg.get("weighted_alpha", ret_cfg.get("hybrid_alpha", 0.5))),
            agreement_bonus_enabled=bool(ret_cfg.get("agreement_bonus_enabled", False)),
            agreement_bonus=float(ret_cfg.get("agreement_bonus", 0.0)),
            multi_hop_enabled=bool(ret_cfg.get("multi_hop_enabled", False)),
            multi_hop_top_seed_hits=int(ret_cfg.get("multi_hop_top_seed_hits", 2)),
            multi_hop_max_entities=int(ret_cfg.get("multi_hop_max_entities", 4)),
            multi_hop_merge_rrf_k=int(ret_cfg.get("multi_hop_merge_rrf_k", 60)),
            multi_hop_mode=str(ret_cfg.get("multi_hop_mode", "hybrid")),
            multi_hop_top_k_dense=(
                int(ret_cfg.get("multi_hop_top_k_dense"))
                if ret_cfg.get("multi_hop_top_k_dense", None) is not None
                else None
            ),
            multi_hop_top_k_bm25=(
                int(ret_cfg.get("multi_hop_top_k_bm25"))
                if ret_cfg.get("multi_hop_top_k_bm25", None) is not None
                else None
            ),
            multi_hop_gate_enabled=bool(ret_cfg.get("multi_hop_gate_enabled", False)),
            multi_hop_gate_overlap_threshold=float(ret_cfg.get("multi_hop_gate_overlap_threshold", 0.35)),
            multi_hop_gate_margin_threshold=float(ret_cfg.get("multi_hop_gate_margin_threshold", 0.01)),
        )
        rerank_trace: dict[str, Any] = {"t_rerank_total_ms": 0.0}
        if self.reranker and bool(rer_cfg.get("enabled", False)):
            hits, rerank_trace = self.reranker.rerank_with_trace(
                question,
                candidates=hits[: int(rer_cfg["top_k_in"])],
                top_k_out=int(rer_cfg["top_k_out"]),
            )
        trace["retrieval_final_topk_ids"] = [h.item_id for h in hits]
        trace["retrieval_final_topk_scores"] = [float(h.score) for h in hits]
        return hits, trace, rerank_trace

    def retrieve(self, question: str, cfg: dict) -> list[RetrievedItem]:
        hits, _, _ = self.retrieve_with_trace(question, cfg)
        return hits

    def answer(self, question: str, cfg: dict) -> PipelineOutput:
        t_total0 = time.perf_counter()
        hits, ret_trace, rerank_trace = self.retrieve_with_trace(question, cfg)
        context_budgeting_trace = {
            "enabled": bool(self.context_budgeting_enabled),
            "strategy": str(self.context_budgeting_strategy),
            "k_eff": int(min(len(hits), int(cfg.get("retrieval", {}).get("top_k_final", len(hits))))),
            "context_budget_tokens": 0,
            "context_tokens_used": 0,
            "policy_branch": "disabled",
            "margin_value": None,
            "agreement_value": None,
            "margin_source": "none",
            "margin_threshold": float(self.context_budget_margin_threshold),
            "agreement_threshold": float(cfg.get("context_budgeting", {}).get("agreement_threshold", 0.35)),
            "keep_full_count": 0,
            "fallback_to_high": False,
            "fallback_reason": "",
            "context_chunk_ids_used": [],
            "context_doc_ids_used": [],
            "context_doc_internal_ids_used": [],
            "redundancy_avg_similarity": 0.0,
            "redundancy_max_similarity": 0.0,
            "aliases_used": dict(self.context_budgeting_resolved.get("aliases_used", {})),
            "unknown_keys": list(self.context_budgeting_resolved.get("unknown_keys", [])),
            "v2_resolved_params": (
                dict(self.context_budgeting_resolved) if self.context_budgeting_strategy == "v2_evidence_first" else {}
            ),
        }
        context_budgeting_trace.update(self.context_budget_threshold_meta)
        latency_ms = {
            "t_retrieval_total_ms": float(ret_trace.get("t_retrieval_total_ms", 0.0)),
            "t_bm25_search_ms": float(ret_trace.get("bm25", {}).get("t_bm25_search_ms", 0.0)),
            "t_query_embed_ms": float(ret_trace.get("dense", {}).get("t_query_embed_ms", 0.0)),
            "t_vector_search_ms": float(ret_trace.get("dense", {}).get("t_vector_search_ms", 0.0)),
            "t_merge_hybrid_ms": float(ret_trace.get("fusion", {}).get("t_merge_hybrid_ms", 0.0)),
            "t_rerank_total_ms": float(rerank_trace.get("t_rerank_total_ms", 0.0)),
        }

        if not self.enable_llm or self.generator is None:
            latency_ms["t_total_ms"] = float((time.perf_counter() - t_total0) * 1000.0)
            return PipelineOutput(
                retrieved=hits,
                retrieval_stages=ret_trace,
                latency_ms=latency_ms,
                context_budgeting=context_budgeting_trace,
                sp3=self.runtime_profile,
            )

        llm_cfg = cfg["llm"]
        if self.context_budgeting_enabled:
            runtime_state = None
            if self.context_budgeting_strategy in {"incremental_sc", "agnostic_acb_sc"}:
                runtime_state = {
                    "query_index": int(self.sc_query_count),
                    "ewma_prefill_ms_per_token": self.sc_ewma_prefill_ms_per_token,
                }
            contexts, t_context_pack_ms, context_budgeting_trace = adaptive_pack_contexts(
                question=question,
                hits=hits,
                retrieval_stages=ret_trace,
                cfg=cfg,
                margin_threshold=float(self.context_budget_margin_threshold),
                resolved_context_budget=self.context_budgeting_resolved,
                runtime_state=runtime_state,
            )
            context_budgeting_trace.update(self.context_budget_threshold_meta)
        else:
            contexts, t_context_pack_ms = self._pack_contexts([h.text for h in hits], cfg)
            used_tokens = int(sum(len(c.split()) for c in contexts))
            context_budgeting_trace["k_eff"] = int(len(contexts))
            context_budgeting_trace["context_tokens_used"] = int(used_tokens)
            context_budgeting_trace["context_budget_tokens"] = int(used_tokens)
            selected_hits = hits[: len(contexts)]
            context_budgeting_trace["context_chunk_ids_used"] = [str(h.item_id) for h in selected_hits]
            doc_ids = [str(h.doc_id) for h in selected_hits]
            dedup_doc_ids: list[str] = []
            seen_doc_ids: set[str] = set()
            for doc_id in doc_ids:
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                dedup_doc_ids.append(doc_id)
            context_budgeting_trace["context_doc_internal_ids_used"] = dedup_doc_ids
            context_budgeting_trace["context_doc_ids_used"] = dedup_doc_ids
        gen = self.generator.generate(
            question=question,
            contexts=contexts,
            temperature=float(llm_cfg["temperature"]),
            top_p=float(llm_cfg["top_p"]),
            max_new_tokens=int(llm_cfg["max_new_tokens"]),
            repeat_penalty=float(llm_cfg["repeat_penalty"]),
            prompt_mode=str(llm_cfg.get("prompt_mode", "rag_strict")),
            direct_template=str(llm_cfg.get("direct_prompt_template", "")),
            answer_postprocess_mode=str(llm_cfg.get("answer_postprocess_mode", "basic")),
            enable_stream_timing=bool(llm_cfg.get("stream_timing", True)),
        )
        t_post0 = time.perf_counter()
        t_postprocess_ms = (time.perf_counter() - t_post0) * 1000.0
        latency_ms.update(
            {
                "t_context_pack_ms": float(t_context_pack_ms),
                "t_prompt_build_ms": float(gen.t_prompt_build_ms),
                "t_llm_total_ms": float(gen.t_llm_total_ms),
                "t_prefill_ms": float(gen.t_prefill_ms),
                "ttft_ms": float(gen.ttft_ms),
                "t_decode_total_ms": float(gen.t_decode_total_ms),
                "t_postprocess_ms": float(t_postprocess_ms),
            }
        )
        latency_ms["t_total_ms"] = float((time.perf_counter() - t_total0) * 1000.0)

        if self.context_budgeting_enabled and self.context_budgeting_strategy in {"incremental_sc", "agnostic_acb_sc"}:
            observed_context_tokens = float(context_budgeting_trace.get("context_tokens_used", 0.0) or 0.0)
            if observed_context_tokens > 0.0:
                observed = float(gen.t_prefill_ms) / max(1.0, observed_context_tokens)
                ewma_alpha = max(0.0, min(1.0, float(cfg.get("context_budgeting", {}).get("ewma_alpha", 0.2))))
                if self.sc_ewma_prefill_ms_per_token is None:
                    self.sc_ewma_prefill_ms_per_token = float(observed)
                else:
                    prev = float(self.sc_ewma_prefill_ms_per_token)
                    self.sc_ewma_prefill_ms_per_token = float((ewma_alpha * observed) + ((1.0 - ewma_alpha) * prev))
            self.sc_query_count += 1

        post_context = {
            "context_chunk_ids_used": list(context_budgeting_trace.get("context_chunk_ids_used", [])),
            "context_doc_ids_used": list(context_budgeting_trace.get("context_doc_ids_used", [])),
            "context_doc_internal_ids_used": list(context_budgeting_trace.get("context_doc_internal_ids_used", [])),
            "keep_full_count": int(context_budgeting_trace.get("keep_full_count", 0) or 0),
            "fallback_to_high": bool(context_budgeting_trace.get("fallback_to_high", False)),
            "fallback_reason": str(context_budgeting_trace.get("fallback_reason", "")),
            "redundancy_avg_similarity": float(context_budgeting_trace.get("redundancy_avg_similarity", 0.0) or 0.0),
            "redundancy_max_similarity": float(context_budgeting_trace.get("redundancy_max_similarity", 0.0) or 0.0),
            "mmr_lambda_max_used": float(context_budgeting_trace.get("mmr_lambda_max_used", 0.0) or 0.0),
        }

        return PipelineOutput(
            retrieved=hits,
            answer=gen.answer,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            total_tokens=gen.total_tokens,
            retrieval_stages=ret_trace,
            latency_ms=latency_ms,
            token_stats={
                "context_tokens": float(gen.prompt_tokens),
                "output_tokens": float(gen.completion_tokens),
                "total_tokens": float(gen.total_tokens),
                "tokens_per_second_decode": float(gen.tokens_per_second_decode),
                "tokens_per_second_prefill": float(gen.tokens_per_second_prefill),
            },
            context_budgeting=context_budgeting_trace,
            post_context=post_context,
            sp3=self.runtime_profile,
        )
