from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from .types import Chunk, Document, RetrievedItem

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
_ENTITY_STOPWORDS = {
    "The",
    "A",
    "An",
    "In",
    "On",
    "At",
    "Of",
    "And",
    "Or",
    "By",
    "For",
    "To",
    "From",
    "With",
    "Without",
    "After",
    "Before",
    "During",
    "What",
    "Which",
    "Who",
    "Where",
    "When",
    "Why",
    "How",
}
_EMBED_CACHE: dict[str, np.ndarray] = {}
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


@dataclass(slots=True)
class IndexItem:
    item_id: str
    doc_id: str
    text: str


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def docs_to_items(docs: dict[str, Document]) -> dict[str, IndexItem]:
    return {
        doc_id: IndexItem(item_id=doc_id, doc_id=doc_id, text=doc.text)
        for doc_id, doc in docs.items()
        if doc.text.strip()
    }


def chunks_to_items(chunks: dict[str, Chunk]) -> dict[str, IndexItem]:
    return {
        cid: IndexItem(item_id=cid, doc_id=chunk.doc_id, text=chunk.text)
        for cid, chunk in chunks.items()
        if chunk.text.strip()
    }


class BM25Retriever:
    def __init__(self, items: dict[str, IndexItem]):
        self.items = items
        self.item_ids = list(items.keys())
        tokenized = [tokenize(items[i].text) for i in self.item_ids]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _trace_from_hits(hits: list[RetrievedItem], latency_ms: float) -> dict[str, object]:
        return {
            "topN_ids": [h.item_id for h in hits],
            "topN_scores": [float(h.score) for h in hits],
            "t_bm25_search_ms": float(latency_ms),
        }

    def search_with_trace(self, query: str, top_k: int) -> tuple[list[RetrievedItem], dict[str, object]]:
        t0 = time.perf_counter()
        q_toks = tokenize(query)
        if not q_toks:
            return [], self._trace_from_hits([], 0.0)
        scores = self.bm25.get_scores(q_toks)
        if len(scores) == 0:
            return [], self._trace_from_hits([], 0.0)
        k = min(top_k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        out: list[RetrievedItem] = []
        for i in idx:
            item = self.items[self.item_ids[int(i)]]
            out.append(
                RetrievedItem(
                    item_id=item.item_id,
                    text=item.text,
                    score=float(scores[int(i)]),
                    doc_id=item.doc_id,
                    source="bm25",
                )
            )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return out, self._trace_from_hits(out, lat_ms)

    def search(self, query: str, top_k: int) -> list[RetrievedItem]:
        out, _ = self.search_with_trace(query, top_k)
        return out


class DenseRetriever:
    def __init__(
        self,
        items: dict[str, IndexItem],
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = True,
        batch_size: int = 64,
        persist_cache: bool = False,
        cache_dir: str = "",
    ):
        self.items = items
        self.item_ids = list(items.keys())
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize = normalize
        self.persist_cache = bool(persist_cache)
        self.cache_dir = str(cache_dir)
        if model_name in _MODEL_CACHE:
            self.model = _MODEL_CACHE[model_name]
        else:
            self.model = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = self.model

        cache_key = self._build_cache_key(model_name, self.item_ids, self.items, passage_prefix, normalize)
        cache_path = self._embedding_cache_path(cache_key)
        if cache_key in _EMBED_CACHE:
            self.doc_emb = _EMBED_CACHE[cache_key]
        elif cache_path is not None and cache_path.exists():
            self.doc_emb = np.load(cache_path, allow_pickle=False).astype(np.float32)
            _EMBED_CACHE[cache_key] = self.doc_emb
        else:
            texts = [f"{passage_prefix}{items[i].text}" for i in self.item_ids]
            self.doc_emb = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            ).astype(np.float32)
            _EMBED_CACHE[cache_key] = self.doc_emb
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, self.doc_emb)

    def _embedding_cache_path(self, cache_key: str) -> Path | None:
        if not self.persist_cache or not self.cache_dir.strip():
            return None
        return Path(self.cache_dir) / f"{cache_key}.npy"

    @staticmethod
    def _trace_from_hits(
        hits: list[RetrievedItem],
        t_query_embed_ms: float,
        t_vector_search_ms: float,
    ) -> dict[str, object]:
        return {
            "topN_ids": [h.item_id for h in hits],
            "topN_scores": [float(h.score) for h in hits],
            "t_query_embed_ms": float(t_query_embed_ms),
            "t_vector_search_ms": float(t_vector_search_ms),
        }

    @staticmethod
    def _build_cache_key(
        model_name: str,
        item_ids: list[str],
        items: dict[str, IndexItem],
        passage_prefix: str,
        normalize: bool,
    ) -> str:
        h = hashlib.sha1()
        h.update(model_name.encode("utf-8"))
        h.update(passage_prefix.encode("utf-8"))
        h.update(str(normalize).encode("utf-8"))
        for item_id in item_ids:
            h.update(item_id.encode("utf-8"))
            h.update(hashlib.sha1(items[item_id].text.encode("utf-8")).digest())
        return h.hexdigest()

    def search_with_trace(self, query: str, top_k: int) -> tuple[list[RetrievedItem], dict[str, object]]:
        t_embed0 = time.perf_counter()
        query_vec = self.model.encode(
            [f"{self.query_prefix}{query}"],
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        ).astype(np.float32)[0]
        t_query_embed_ms = (time.perf_counter() - t_embed0) * 1000.0

        t_search0 = time.perf_counter()
        scores = self.doc_emb @ query_vec
        k = min(top_k, len(scores))
        if k <= 0:
            return [], self._trace_from_hits([], t_query_embed_ms, 0.0)
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        out: list[RetrievedItem] = []
        for i in idx:
            item = self.items[self.item_ids[int(i)]]
            out.append(
                RetrievedItem(
                    item_id=item.item_id,
                    text=item.text,
                    score=float(scores[int(i)]),
                    doc_id=item.doc_id,
                    source="dense",
                )
            )
        t_vector_search_ms = (time.perf_counter() - t_search0) * 1000.0
        return out, self._trace_from_hits(out, t_query_embed_ms, t_vector_search_ms)

    def search(self, query: str, top_k: int) -> list[RetrievedItem]:
        out, _ = self.search_with_trace(query, top_k)
        return out


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever | None,
        dense: DenseRetriever | None,
        alpha: float = 0.5,
        fusion_method: str = "RRF",
        rrf_k: int = 60,
        normalize_scores: bool = True,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha
        self.fusion_method = fusion_method.upper()
        self.rrf_k = int(rrf_k)
        self.normalize_scores = normalize_scores

    @staticmethod
    def _minmax(x: dict[str, float]) -> dict[str, float]:
        if not x:
            return {}
        vals = list(x.values())
        lo, hi = min(vals), max(vals)
        if math.isclose(lo, hi):
            return {k: 1.0 for k in x}
        return {k: (v - lo) / (hi - lo) for k, v in x.items()}

    @staticmethod
    def _fuse_rrf(
        dense_hits: list[RetrievedItem],
        bm25_hits: list[RetrievedItem],
        rrf_k: int,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            out[hit.item_id] = out.get(hit.item_id, 0.0) + (1.0 / (rrf_k + rank))
        for rank, hit in enumerate(bm25_hits, start=1):
            out[hit.item_id] = out.get(hit.item_id, 0.0) + (1.0 / (rrf_k + rank))
        return out

    def _fuse_weighted_sum(
        self,
        dense_hits: list[RetrievedItem],
        bm25_hits: list[RetrievedItem],
        alpha: float,
    ) -> dict[str, float]:
        dense_scores = {r.item_id: r.score for r in dense_hits}
        bm25_scores = {r.item_id: r.score for r in bm25_hits}
        if self.normalize_scores:
            dense_scores = self._minmax(dense_scores)
            bm25_scores = self._minmax(bm25_scores)
        all_ids = set(dense_scores) | set(bm25_scores)
        return {
            item_id: float(alpha * dense_scores.get(item_id, 0.0) + (1.0 - alpha) * bm25_scores.get(item_id, 0.0))
            for item_id in all_ids
        }

    @staticmethod
    def _extract_entities(seed_texts: list[str], query: str, max_entities: int) -> list[str]:
        q_lower = query.lower()
        counts: dict[str, int] = {}
        for txt in seed_texts:
            for ent in _ENTITY_RE.findall(txt):
                ent = " ".join(ent.split()).strip()
                if not ent or ent in _ENTITY_STOPWORDS:
                    continue
                if ent.lower() in q_lower:
                    continue
                counts[ent] = counts.get(ent, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))
        return [k for k, _ in ranked[: max(0, int(max_entities))]]

    def _fuse_selected(
        self,
        dense_hits: list[RetrievedItem],
        bm25_hits: list[RetrievedItem],
        selected_fusion: str,
        selected_rrf_k: int,
        selected_alpha: float,
    ) -> dict[str, float]:
        if selected_fusion == "WEIGHTED_SUM":
            return self._fuse_weighted_sum(dense_hits, bm25_hits, selected_alpha)
        return self._fuse_rrf(dense_hits, bm25_hits, selected_rrf_k)

    def search(
        self,
        query: str,
        top_k_dense: int,
        top_k_bm25: int,
        top_k_final: int,
        retriever_mode: str = "hybrid",
        fusion_method: str | None = None,
        rrf_k: int | None = None,
        weighted_alpha: float | None = None,
        agreement_bonus_enabled: bool = False,
        agreement_bonus: float = 0.0,
        multi_hop_enabled: bool = False,
        multi_hop_top_seed_hits: int = 2,
        multi_hop_max_entities: int = 4,
        multi_hop_merge_rrf_k: int = 60,
        multi_hop_mode: str = "hybrid",
        multi_hop_top_k_dense: int | None = None,
        multi_hop_top_k_bm25: int | None = None,
        multi_hop_gate_enabled: bool = False,
        multi_hop_gate_overlap_threshold: float = 0.35,
        multi_hop_gate_margin_threshold: float = 0.01,
    ) -> list[RetrievedItem]:
        out, _ = self.search_with_trace(
            query=query,
            top_k_dense=top_k_dense,
            top_k_bm25=top_k_bm25,
            top_k_final=top_k_final,
            retriever_mode=retriever_mode,
            fusion_method=fusion_method,
            rrf_k=rrf_k,
            weighted_alpha=weighted_alpha,
            agreement_bonus_enabled=agreement_bonus_enabled,
            agreement_bonus=agreement_bonus,
            multi_hop_enabled=multi_hop_enabled,
            multi_hop_top_seed_hits=multi_hop_top_seed_hits,
            multi_hop_max_entities=multi_hop_max_entities,
            multi_hop_merge_rrf_k=multi_hop_merge_rrf_k,
            multi_hop_mode=multi_hop_mode,
            multi_hop_top_k_dense=multi_hop_top_k_dense,
            multi_hop_top_k_bm25=multi_hop_top_k_bm25,
            multi_hop_gate_enabled=multi_hop_gate_enabled,
            multi_hop_gate_overlap_threshold=multi_hop_gate_overlap_threshold,
            multi_hop_gate_margin_threshold=multi_hop_gate_margin_threshold,
        )
        return out

    def search_with_trace(
        self,
        query: str,
        top_k_dense: int,
        top_k_bm25: int,
        top_k_final: int,
        retriever_mode: str = "hybrid",
        fusion_method: str | None = None,
        rrf_k: int | None = None,
        weighted_alpha: float | None = None,
        agreement_bonus_enabled: bool = False,
        agreement_bonus: float = 0.0,
        multi_hop_enabled: bool = False,
        multi_hop_top_seed_hits: int = 2,
        multi_hop_max_entities: int = 4,
        multi_hop_merge_rrf_k: int = 60,
        multi_hop_mode: str = "hybrid",
        multi_hop_top_k_dense: int | None = None,
        multi_hop_top_k_bm25: int | None = None,
        multi_hop_gate_enabled: bool = False,
        multi_hop_gate_overlap_threshold: float = 0.35,
        multi_hop_gate_margin_threshold: float = 0.01,
    ) -> tuple[list[RetrievedItem], dict[str, object]]:
        mode = retriever_mode.lower()
        selected_fusion = (fusion_method or self.fusion_method).upper()
        selected_rrf_k = int(self.rrf_k if rrf_k is None else rrf_k)
        selected_alpha = float(self.alpha if weighted_alpha is None else weighted_alpha)

        dense_hits: list[RetrievedItem] = []
        bm25_hits: list[RetrievedItem] = []
        dense_trace: dict[str, object] = {
            "topN_ids": [],
            "topN_scores": [],
            "t_query_embed_ms": 0.0,
            "t_vector_search_ms": 0.0,
        }
        bm25_trace: dict[str, object] = {
            "topN_ids": [],
            "topN_scores": [],
            "t_bm25_search_ms": 0.0,
        }

        if mode in {"hybrid", "dense_only"}:
            if self.dense is None:
                raise RuntimeError("dense retriever is not initialized for mode requiring dense retrieval")
            dense_hits, dense_trace = self.dense.search_with_trace(query, top_k_dense)
        if mode in {"hybrid", "bm25_only"}:
            if self.bm25 is None:
                raise RuntimeError("bm25 retriever is not initialized for mode requiring bm25 retrieval")
            bm25_hits, bm25_trace = self.bm25.search_with_trace(query, top_k_bm25)

        dense_ids = set([h.item_id for h in dense_hits])
        bm25_ids = set([h.item_id for h in bm25_hits])
        overlap_ids = dense_ids & bm25_ids
        overlap_count = len(overlap_ids)
        overlap_ratio = float(overlap_count / max(1, min(len(dense_ids), len(bm25_ids)))) if dense_ids and bm25_ids else 0.0

        id_to_item: dict[str, RetrievedItem] = {}
        for hit in dense_hits:
            id_to_item[hit.item_id] = hit
        for hit in bm25_hits:
            if hit.item_id not in id_to_item:
                id_to_item[hit.item_id] = hit

        multi_hop_trace: dict[str, object] = {
            "enabled": bool(multi_hop_enabled and mode == "hybrid"),
            "used": False,
            "mode": str(multi_hop_mode).lower(),
            "entities": [],
            "expanded_query": "",
            "merge_rrf_k": int(multi_hop_merge_rrf_k),
            "gate_enabled": bool(multi_hop_gate_enabled),
            "gate_triggered": True,
            "gate_overlap_threshold": float(multi_hop_gate_overlap_threshold),
            "gate_margin_threshold": float(multi_hop_gate_margin_threshold),
            "first_pass_margin": 0.0,
            "first_pass_overlap_ratio": float(overlap_ratio),
            "top_k_dense": int(top_k_dense if multi_hop_top_k_dense is None else multi_hop_top_k_dense),
            "top_k_bm25": int(top_k_bm25 if multi_hop_top_k_bm25 is None else multi_hop_top_k_bm25),
            "dense_topN_ids": [],
            "dense_topN_scores": [],
            "bm25_topN_ids": [],
            "bm25_topN_scores": [],
            "t_hop2_query_embed_ms": 0.0,
            "t_hop2_vector_search_ms": 0.0,
            "t_hop2_bm25_search_ms": 0.0,
            "t_hop2_merge_ms": 0.0,
            "t_hop2_total_ms": 0.0,
        }

        t_merge0 = time.perf_counter()
        fused_scores: dict[str, float]
        source = "hybrid"
        if mode == "dense_only":
            fused_scores = {h.item_id: float(h.score) for h in dense_hits}
            source = "dense"
        elif mode == "bm25_only":
            fused_scores = {h.item_id: float(h.score) for h in bm25_hits}
            source = "bm25"
        else:
            fused_scores = self._fuse_selected(
                dense_hits=dense_hits,
                bm25_hits=bm25_hits,
                selected_fusion=selected_fusion,
                selected_rrf_k=selected_rrf_k,
                selected_alpha=selected_alpha,
            )
            source = "hybrid_weighted_sum" if selected_fusion == "WEIGHTED_SUM" else "hybrid_rrf"
        agreement_bonus_abs = 0.0
        agreement_boosted_count = 0
        if mode == "hybrid" and bool(agreement_bonus_enabled) and float(agreement_bonus) > 0.0 and fused_scores:
            base_scale = max(float(v) for v in fused_scores.values())
            if not math.isfinite(base_scale) or base_scale <= 0.0:
                base_scale = 1.0
            agreement_bonus_abs = float(base_scale * float(agreement_bonus))
            for item_id in overlap_ids:
                if item_id in fused_scores:
                    fused_scores[item_id] = float(fused_scores[item_id] + agreement_bonus_abs)
                    agreement_boosted_count += 1

        should_run_hop2 = bool(multi_hop_enabled) and mode == "hybrid" and bool(fused_scores)
        if should_run_hop2 and bool(multi_hop_gate_enabled):
            ranked_now = sorted(fused_scores.items(), key=lambda x: (-float(x[1]), str(x[0])))
            margin = 0.0
            if len(ranked_now) >= 2:
                margin = float(ranked_now[0][1]) - float(ranked_now[1][1])
            multi_hop_trace["first_pass_margin"] = float(margin)
            multi_hop_trace["first_pass_overlap_ratio"] = float(overlap_ratio)
            gate_triggered = bool(
                float(overlap_ratio) < float(multi_hop_gate_overlap_threshold)
                or float(margin) < float(multi_hop_gate_margin_threshold)
            )
            multi_hop_trace["gate_triggered"] = bool(gate_triggered)
            should_run_hop2 = bool(gate_triggered)

        if should_run_hop2:
            seed_ranked = sorted(fused_scores.items(), key=lambda x: (-float(x[1]), str(x[0])))
            seed_texts: list[str] = []
            for item_id, _ in seed_ranked[: max(1, int(multi_hop_top_seed_hits))]:
                base = id_to_item.get(item_id)
                if base is not None and base.text.strip():
                    seed_texts.append(base.text)
            entities = self._extract_entities(seed_texts, query=query, max_entities=int(multi_hop_max_entities))
            if entities:
                expanded_query = (query.strip() + " " + " ".join(entities)).strip()
                hop2_t0 = time.perf_counter()
                hop2_mode = str(multi_hop_mode).lower()
                run_h2_dense = hop2_mode in {"hybrid", "dense_only"}
                run_h2_bm25 = hop2_mode in {"hybrid", "bm25_only"}
                dense_hits_h2: list[RetrievedItem] = []
                bm25_hits_h2: list[RetrievedItem] = []
                dense_trace_h2: dict[str, object] = {
                    "topN_ids": [],
                    "topN_scores": [],
                    "t_query_embed_ms": 0.0,
                    "t_vector_search_ms": 0.0,
                }
                bm25_trace_h2: dict[str, object] = {
                    "topN_ids": [],
                    "topN_scores": [],
                    "t_bm25_search_ms": 0.0,
                }
                if run_h2_dense and self.dense is not None:
                    hop2_k_dense = int(top_k_dense if multi_hop_top_k_dense is None else multi_hop_top_k_dense)
                    dense_hits_h2, dense_trace_h2 = self.dense.search_with_trace(expanded_query, hop2_k_dense)
                if run_h2_bm25 and self.bm25 is not None:
                    hop2_k_bm25 = int(top_k_bm25 if multi_hop_top_k_bm25 is None else multi_hop_top_k_bm25)
                    bm25_hits_h2, bm25_trace_h2 = self.bm25.search_with_trace(expanded_query, hop2_k_bm25)
                for hit in dense_hits_h2:
                    if hit.item_id not in id_to_item:
                        id_to_item[hit.item_id] = hit
                for hit in bm25_hits_h2:
                    if hit.item_id not in id_to_item:
                        id_to_item[hit.item_id] = hit
                hop2_merge_t0 = time.perf_counter()
                if hop2_mode == "dense_only":
                    fused_scores_h2 = {h.item_id: float(h.score) for h in dense_hits_h2}
                elif hop2_mode == "bm25_only":
                    fused_scores_h2 = {h.item_id: float(h.score) for h in bm25_hits_h2}
                else:
                    fused_scores_h2 = self._fuse_selected(
                        dense_hits=dense_hits_h2,
                        bm25_hits=bm25_hits_h2,
                        selected_fusion=selected_fusion,
                        selected_rrf_k=selected_rrf_k,
                        selected_alpha=selected_alpha,
                    )
                rank1 = [item_id for item_id, _ in seed_ranked]
                rank2 = [item_id for item_id, _ in sorted(fused_scores_h2.items(), key=lambda x: (-float(x[1]), str(x[0])))]
                merged_scores: dict[str, float] = {}
                merge_rrf_k = max(1, int(multi_hop_merge_rrf_k))
                for rank, item_id in enumerate(rank1, start=1):
                    merged_scores[item_id] = merged_scores.get(item_id, 0.0) + (1.0 / (merge_rrf_k + rank))
                for rank, item_id in enumerate(rank2, start=1):
                    merged_scores[item_id] = merged_scores.get(item_id, 0.0) + (1.0 / (merge_rrf_k + rank))
                if merged_scores:
                    fused_scores = merged_scores
                multi_hop_trace = {
                    "enabled": True,
                    "used": True,
                    "mode": hop2_mode,
                    "entities": entities,
                    "expanded_query": expanded_query,
                    "merge_rrf_k": int(merge_rrf_k),
                    "top_k_dense": int(top_k_dense if multi_hop_top_k_dense is None else multi_hop_top_k_dense),
                    "top_k_bm25": int(top_k_bm25 if multi_hop_top_k_bm25 is None else multi_hop_top_k_bm25),
                    "dense_topN_ids": [h.item_id for h in dense_hits_h2],
                    "dense_topN_scores": [float(h.score) for h in dense_hits_h2],
                    "bm25_topN_ids": [h.item_id for h in bm25_hits_h2],
                    "bm25_topN_scores": [float(h.score) for h in bm25_hits_h2],
                    "t_hop2_query_embed_ms": float(dense_trace_h2.get("t_query_embed_ms", 0.0)),
                    "t_hop2_vector_search_ms": float(dense_trace_h2.get("t_vector_search_ms", 0.0)),
                    "t_hop2_bm25_search_ms": float(bm25_trace_h2.get("t_bm25_search_ms", 0.0)),
                    "t_hop2_merge_ms": float((time.perf_counter() - hop2_merge_t0) * 1000.0),
                    "t_hop2_total_ms": float((time.perf_counter() - hop2_t0) * 1000.0),
                }
            else:
                multi_hop_trace = {
                    **multi_hop_trace,
                    "enabled": True,
                    "used": False,
                }
        elif bool(multi_hop_enabled) and mode == "hybrid":
            multi_hop_trace = {
                **multi_hop_trace,
                "enabled": True,
                "used": False,
            }

        t_merge_hybrid_ms = (time.perf_counter() - t_merge0) * 1000.0

        fused = sorted(fused_scores.items(), key=lambda x: (-float(x[1]), str(x[0])))[:top_k_final]
        out: list[RetrievedItem] = []
        for item_id, score in fused:
            base = id_to_item.get(item_id)
            if base is None:
                # Fall back to dense index metadata (shared ids).
                if self.dense is not None:
                    item = self.dense.items[item_id]
                elif self.bm25 is not None:
                    item = self.bm25.items[item_id]
                else:
                    raise RuntimeError("No retriever index is available to recover item metadata")
                out.append(
                    RetrievedItem(
                        item_id=item_id,
                        text=item.text,
                        score=float(score),
                        doc_id=item.doc_id,
                        source=source,
                    )
                )
            else:
                out.append(
                    RetrievedItem(
                        item_id=base.item_id,
                        text=base.text,
                        score=float(score),
                        doc_id=base.doc_id,
                        source=source,
                    )
                )

        hop2_extra_ms = (
            float(multi_hop_trace.get("t_hop2_query_embed_ms", 0.0))
            + float(multi_hop_trace.get("t_hop2_vector_search_ms", 0.0))
            + float(multi_hop_trace.get("t_hop2_bm25_search_ms", 0.0))
        )
        t_retrieval_total_ms = (
            float(dense_trace.get("t_query_embed_ms", 0.0))
            + float(dense_trace.get("t_vector_search_ms", 0.0))
            + float(bm25_trace.get("t_bm25_search_ms", 0.0))
            + float(hop2_extra_ms)
            + float(t_merge_hybrid_ms)
        )

        trace: dict[str, object] = {
            "retriever_mode": mode,
            "fusion_method": selected_fusion if mode == "hybrid" else None,
            "rrf_k": selected_rrf_k if mode == "hybrid" and selected_fusion != "WEIGHTED_SUM" else None,
            "weighted_alpha": selected_alpha if mode == "hybrid" and selected_fusion == "WEIGHTED_SUM" else None,
            "agreement_bonus_enabled": bool(mode == "hybrid" and agreement_bonus_enabled),
            "agreement_bonus": float(agreement_bonus) if mode == "hybrid" else 0.0,
            "agreement_bonus_abs": float(agreement_bonus_abs),
            "agreement_boosted_count": int(agreement_boosted_count),
            "bm25": bm25_trace,
            "dense": dense_trace,
            "fusion": {
                "fused_topK_ids": [h.item_id for h in out],
                "fused_topK_scores": [float(h.score) for h in out],
                "t_merge_hybrid_ms": float(t_merge_hybrid_ms),
            },
            "multi_hop": multi_hop_trace,
            "overlap_count": int(overlap_count),
            "overlap_ratio": float(overlap_ratio),
            "t_retrieval_total_ms": float(t_retrieval_total_ms),
        }
        return out, trace


def unique_doc_ids(items: Iterable[RetrievedItem]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if item.doc_id in seen:
            continue
        seen.add(item.doc_id)
        ordered.append(item.doc_id)
    return ordered
