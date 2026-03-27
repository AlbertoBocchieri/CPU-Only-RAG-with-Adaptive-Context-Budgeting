from __future__ import annotations

import glob
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from agnostic_cpu_rag.context_controller import ContextController
from agnostic_cpu_rag.records import CoverageGoal, RetrievedCandidate

from .types import RetrievedItem

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_V2_ALIAS_MAP: dict[str, str] = {
    "k_low": "k_eff_floor",
}

_V2_CANONICAL_KEYS: set[str] = {
    "enabled",
    "strategy",
    "keep_full_count",
    "mmr_lambda",
    "min_low_savings_ratio",
    "k_eff_floor",
    "snippet_from_rank",
    "snippet_window_tokens",
    "low_margin_multiplier",
    "low_agreement_multiplier",
    "medium_branch_enabled",
    "medium_budget_tokens",
    "medium_k_eff",
    "budget_low_tokens",
    "budget_high_tokens",
    "max_chunks_hard_cap",
    "margin_threshold",
    "margin_threshold_quantile",
    "margin_threshold_stage2_glob",
    "margin_threshold_stage2_retriever_mode",
    "margin_threshold_fallback",
    "agreement_threshold",
    "use_rerank_scores_if_available",
    "top_doc_saliency_tokens",
    "saliency_entity_weight",
    "dynamic_mmr_enabled",
    "dynamic_mmr_threshold",
    "dynamic_mmr_boost",
    "dynamic_mmr_cap",
}

# Keys that might come from legacy base configs while v2 strategy is enabled.
_V2_LEGACY_ALLOWED_KEYS: set[str] = {
    "packing_mode_low",
    "packing_mode_high",
}


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    qq = min(1.0, max(0.0, float(q)))
    idx = int(round((len(s) - 1) * qq))
    idx = min(len(s) - 1, max(0, idx))
    return float(s[idx])


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(float(v) for v in values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    med = _median(values)
    deviations = [abs(float(v) - med) for v in values]
    return float(_median(deviations))


def _score_margin(scores: list[Any]) -> float | None:
    clean = [v for v in (_safe_float(x) for x in scores) if v is not None]
    if len(clean) >= 2:
        return float(clean[0] - clean[1])
    if len(clean) == 1:
        return float(clean[0])
    return None


def _extract_margin_from_row(row: dict[str, Any]) -> float | None:
    stages = row.get("retrieval_stages", {})
    final_scores = stages.get("retrieval_final_topk_scores", [])
    margin = _score_margin(final_scores)
    if margin is not None:
        return margin
    fused_scores = stages.get("fusion", {}).get("fused_topK_scores", [])
    return _score_margin(fused_scores)


def estimate_margin_threshold_from_stage2(
    stage2_glob: str,
    quantile: float,
    fallback: float,
    retriever_mode_filter: str = "hybrid",
) -> tuple[float, dict[str, Any]]:
    paths = sorted(glob.glob(stage2_glob))
    margins: list[float] = []
    for p in paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    mode = str(row.get("retrieval_stages", {}).get("retriever_mode", "")).lower()
                    if retriever_mode_filter and mode != retriever_mode_filter.lower():
                        continue
                    m = _extract_margin_from_row(row)
                    if m is not None:
                        margins.append(float(m))
        except Exception:
            continue
    if not margins:
        return float(fallback), {
            "margin_threshold_source": "fallback",
            "margin_threshold_quantile": float(quantile),
            "margin_threshold_stage2_glob": stage2_glob,
            "margin_threshold_retriever_mode": retriever_mode_filter,
            "margin_threshold_samples": 0,
        }
    value = _quantile(margins, quantile)
    return float(value), {
        "margin_threshold_source": "stage2_quantile",
        "margin_threshold_quantile": float(quantile),
        "margin_threshold_stage2_glob": stage2_glob,
        "margin_threshold_retriever_mode": retriever_mode_filter,
        "margin_threshold_samples": int(len(margins)),
    }


def resolve_margin_threshold(context_budget_cfg: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    explicit = context_budget_cfg.get("margin_threshold", None)
    parsed = _safe_float(explicit)
    if parsed is not None:
        return float(parsed), {
            "margin_threshold_source": "explicit",
            "margin_threshold_quantile": float(context_budget_cfg.get("margin_threshold_quantile", 0.9)),
            "margin_threshold_stage2_glob": str(context_budget_cfg.get("margin_threshold_stage2_glob", "")),
            "margin_threshold_retriever_mode": str(
                context_budget_cfg.get("margin_threshold_stage2_retriever_mode", "hybrid")
            ),
            "margin_threshold_samples": 0,
        }
    quantile = float(context_budget_cfg.get("margin_threshold_quantile", 0.9))
    fallback = float(context_budget_cfg.get("margin_threshold_fallback", 0.003))
    stage2_glob = str(context_budget_cfg.get("margin_threshold_stage2_glob", "results/*/stage2/*/per_query.jsonl"))
    retriever_mode_filter = str(context_budget_cfg.get("margin_threshold_stage2_retriever_mode", "hybrid"))
    return estimate_margin_threshold_from_stage2(
        stage2_glob=stage2_glob,
        quantile=quantile,
        fallback=fallback,
        retriever_mode_filter=retriever_mode_filter,
    )


def _question_token_set(question: str) -> set[str]:
    return set(_WORD_RE.findall(question.lower()))


def _question_entity_set(question: str) -> set[str]:
    entities: set[str] = set()
    for tok in re.findall(r"\b[A-Z][A-Za-z0-9_\-]+\b", str(question)):
        entities.add(tok.lower())
    return entities


def _text_token_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _word_count(text: str) -> int:
    return len(text.split())


def _leading_window(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text.strip()
    return " ".join(words[:max_tokens]).strip()


def _sentence_split(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip())]
    return [p for p in parts if p]


def _sentence_overlap_score(sentence: str, q_tokens: set[str]) -> int:
    if not q_tokens:
        return 0
    score = 0
    for tok in _WORD_RE.findall(sentence.lower()):
        if tok in q_tokens:
            score += 1
    return score


def _snippet_window(text: str, question: str, window_tokens: int) -> str:
    window_tokens = max(16, int(window_tokens))
    words = text.split()
    if len(words) <= window_tokens:
        return text.strip()

    sentences = _sentence_split(text)
    if len(sentences) <= 1:
        return _leading_window(text, window_tokens)

    q_tokens = _question_token_set(question)
    scored = [(idx, _sentence_overlap_score(sent, q_tokens)) for idx, sent in enumerate(sentences)]
    scored.sort(key=lambda x: (x[1], -x[0]), reverse=True)
    best_idx = int(scored[0][0])
    if scored[0][1] <= 0:
        return _leading_window(text, window_tokens)

    left = best_idx
    right = best_idx
    selected = [sentences[best_idx]]
    used = _word_count(selected[0])
    while used < window_tokens and (left > 0 or right < len(sentences) - 1):
        next_left = left - 1
        next_right = right + 1
        cand_left = sentences[next_left] if next_left >= 0 else ""
        cand_right = sentences[next_right] if next_right < len(sentences) else ""
        left_score = _sentence_overlap_score(cand_left, q_tokens) if cand_left else -1
        right_score = _sentence_overlap_score(cand_right, q_tokens) if cand_right else -1
        if right_score > left_score and cand_right:
            selected.append(cand_right)
            right = next_right
            used += _word_count(cand_right)
        elif cand_left:
            selected.insert(0, cand_left)
            left = next_left
            used += _word_count(cand_left)
        elif cand_right:
            selected.append(cand_right)
            right = next_right
            used += _word_count(cand_right)
        else:
            break
    return _leading_window(" ".join(selected), window_tokens)


def _saliency_snippet_with_title(
    text: str,
    question: str,
    doc_id: str,
    budget_tokens: int,
    entity_weight: float,
) -> str:
    budget_tokens = max(24, int(budget_tokens))
    text = str(text or "").strip()
    if not text:
        return ""
    q_tokens = _question_token_set(question)
    q_entities = _question_entity_set(question)
    sentences = _sentence_split(text)
    if not sentences:
        return ""
    scored: list[tuple[float, int, int, str]] = []
    for idx, sent in enumerate(sentences):
        stoks = _text_token_set(sent)
        tok_overlap = len(stoks.intersection(q_tokens))
        entity_overlap = len(stoks.intersection(q_entities))
        score = float(tok_overlap + (float(entity_weight) * float(entity_overlap)))
        scored.append((score, tok_overlap, entity_overlap, sent))
    scored.sort(key=lambda x: (x[0], x[1], x[2], len(x[3])), reverse=True)
    chosen: list[str] = []
    used = 0
    for _, _, _, sent in scored:
        s = str(sent).strip()
        if not s:
            continue
        wc = _word_count(s)
        if wc <= 0:
            continue
        if used + wc > budget_tokens:
            remain = budget_tokens - used
            if remain > 8:
                s = _leading_window(s, remain)
                wc = _word_count(s)
            else:
                continue
        if wc <= 0:
            continue
        chosen.append(s)
        used += wc
        if used >= budget_tokens:
            break
    if not chosen:
        return ""
    header = f"[{str(doc_id)}]"
    body = " ".join(chosen)
    return f"{header} {body}".strip()


def _compute_margin(
    retrieval_stages: dict[str, Any],
    use_rerank_scores_if_available: bool,
) -> tuple[float, str]:
    if use_rerank_scores_if_available:
        rerank_scores = retrieval_stages.get("retrieval_final_topk_scores", [])
        margin = _score_margin(rerank_scores)
        if margin is not None:
            return float(margin), "rerank"
    fused_scores = retrieval_stages.get("fusion", {}).get("fused_topK_scores", [])
    margin = _score_margin(fused_scores)
    if margin is not None:
        return float(margin), "fusion"
    return 0.0, "none"


def _score_list(
    retrieval_stages: dict[str, Any],
    use_rerank_scores_if_available: bool,
) -> tuple[list[float], str]:
    if use_rerank_scores_if_available:
        rerank_scores = retrieval_stages.get("retrieval_final_topk_scores", [])
        clean = [v for v in (_safe_float(x) for x in rerank_scores) if v is not None]
        if clean:
            return [float(v) for v in clean], "rerank"
    fused_scores = retrieval_stages.get("fusion", {}).get("fused_topK_scores", [])
    clean = [v for v in (_safe_float(x) for x in fused_scores) if v is not None]
    if clean:
        return [float(v) for v in clean], "fusion"
    return [], "none"


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        v = str(value)
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(inter / union)


def _normalize_relevance(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if math.isclose(lo, hi):
        return [1.0 for _ in scores]
    return [float((s - lo) / (hi - lo)) for s in scores]


def _resolve_alias(
    cb: dict[str, Any],
    canonical_key: str,
    aliases_used: dict[str, str],
) -> Any:
    if canonical_key in cb:
        return cb.get(canonical_key)
    for alias, target in _V2_ALIAS_MAP.items():
        if target == canonical_key and alias in cb:
            aliases_used[alias] = canonical_key
            return cb.get(alias)
    return None


def resolve_context_budgeting_config(context_budget_cfg: dict[str, Any]) -> dict[str, Any]:
    cb = dict(context_budget_cfg or {})
    enabled = bool(cb.get("enabled", False))
    strategy = str(cb.get("strategy", "v1")).strip().lower()
    aliases_used: dict[str, str] = {}

    resolved: dict[str, Any] = {
        "enabled": enabled,
        "strategy": strategy,
        "k_low": max(1, int(cb.get("k_low", 5))),
        "budget_low_tokens": max(64, int(cb.get("budget_low_tokens", 600))),
        "budget_high_tokens": max(64, int(cb.get("budget_high_tokens", 1200))),
        "agreement_threshold": float(cb.get("agreement_threshold", 0.35)),
        "use_rerank_scores_if_available": bool(cb.get("use_rerank_scores_if_available", True)),
        "snippet_window_tokens": max(24, int(cb.get("snippet_window_tokens", 80))),
        "max_chunks_hard_cap": max(1, int(cb.get("max_chunks_hard_cap", 20))),
        "packing_mode_low": str(cb.get("packing_mode_low", "full_or_light")),
        "packing_mode_high": str(cb.get("packing_mode_high", "snippet")),
        "aliases_used": aliases_used,
        "unknown_keys": [],
        "legacy_keys_present": [],
    }
    resolved["budget_high_tokens"] = max(
        resolved["budget_low_tokens"],
        resolved["budget_high_tokens"],
    )

    if strategy != "v2_evidence_first":
        return resolved

    allowed_keys = set(_V2_CANONICAL_KEYS)
    allowed_keys.update(_V2_ALIAS_MAP.keys())
    allowed_keys.update(_V2_LEGACY_ALLOWED_KEYS)
    unknown_keys = sorted(k for k in cb.keys() if k not in allowed_keys)
    if unknown_keys:
        raise ValueError(
            "Unknown context_budgeting keys for strategy=v2_evidence_first: "
            + ", ".join(unknown_keys)
        )
    resolved["unknown_keys"] = list(unknown_keys)
    resolved["legacy_keys_present"] = sorted(k for k in cb.keys() if k in _V2_LEGACY_ALLOWED_KEYS)

    keep_full_count = _resolve_alias(cb, "keep_full_count", aliases_used)
    mmr_lambda = _resolve_alias(cb, "mmr_lambda", aliases_used)
    min_low_savings_ratio = _resolve_alias(cb, "min_low_savings_ratio", aliases_used)
    k_eff_floor = _resolve_alias(cb, "k_eff_floor", aliases_used)
    snippet_from_rank = _resolve_alias(cb, "snippet_from_rank", aliases_used)
    low_margin_multiplier = _resolve_alias(cb, "low_margin_multiplier", aliases_used)
    low_agreement_multiplier = _resolve_alias(cb, "low_agreement_multiplier", aliases_used)
    medium_branch_enabled = _resolve_alias(cb, "medium_branch_enabled", aliases_used)
    medium_budget_tokens = _resolve_alias(cb, "medium_budget_tokens", aliases_used)
    medium_k_eff = _resolve_alias(cb, "medium_k_eff", aliases_used)

    resolved.update(
        {
            "keep_full_count": max(1, int(keep_full_count if keep_full_count is not None else 2)),
            "mmr_lambda": max(0.0, float(mmr_lambda if mmr_lambda is not None else 0.30)),
            "min_low_savings_ratio": max(
                0.0,
                min(1.0, float(min_low_savings_ratio if min_low_savings_ratio is not None else 0.15)),
            ),
            "k_eff_floor": max(1, int(k_eff_floor if k_eff_floor is not None else max(5, resolved["k_low"]))),
            "snippet_from_rank": max(1, int(snippet_from_rank if snippet_from_rank is not None else 3)),
            "low_margin_multiplier": max(
                1.0,
                float(low_margin_multiplier if low_margin_multiplier is not None else 5.0),
            ),
            "low_agreement_multiplier": max(
                1.0,
                float(low_agreement_multiplier if low_agreement_multiplier is not None else 1.0),
            ),
            "medium_branch_enabled": bool(medium_branch_enabled if medium_branch_enabled is not None else True),
            "medium_budget_tokens": max(
                resolved["budget_low_tokens"],
                int(
                    medium_budget_tokens
                    if medium_budget_tokens is not None
                    else (resolved["budget_low_tokens"] + resolved["budget_high_tokens"]) // 2
                ),
            ),
            "medium_k_eff": max(
                1,
                int(medium_k_eff if medium_k_eff is not None else max(resolved["k_low"], 8)),
            ),
            "top_doc_saliency_tokens": max(32, int(cb.get("top_doc_saliency_tokens", 192))),
            "saliency_entity_weight": max(0.0, float(cb.get("saliency_entity_weight", 2.0))),
            "dynamic_mmr_enabled": bool(cb.get("dynamic_mmr_enabled", True)),
            "dynamic_mmr_threshold": max(0.0, min(1.0, float(cb.get("dynamic_mmr_threshold", 0.45)))),
            "dynamic_mmr_boost": max(0.0, float(cb.get("dynamic_mmr_boost", 0.60))),
            "dynamic_mmr_cap": max(0.0, float(cb.get("dynamic_mmr_cap", 1.20))),
        }
    )
    return resolved


def _resolve_probe_runtime_budget(
    cb: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    prefill_target_ms = max(1000.0, float(cb.get("prefill_target_ms", 18000.0)))
    cap_bootstrap_tokens = max(128, int(cb.get("cap_bootstrap_tokens", 1024)))
    cap_min_tokens = max(64, int(cb.get("cap_min_tokens", 768)))
    cap_max_tokens = max(cap_min_tokens, int(cb.get("cap_max_tokens", 1536)))
    fixed_cap_tokens_raw = cb.get("fixed_cap_tokens", None)
    fixed_cap_tokens = None if fixed_cap_tokens_raw is None else max(64, int(fixed_cap_tokens_raw))
    warmup_queries = max(0, int(cb.get("warmup_queries", 8)))
    ewma_prefill_ms_per_token = _safe_float(runtime.get("ewma_prefill_ms_per_token"))
    run_query_index = max(0, int(runtime.get("query_index", 0)))
    if fixed_cap_tokens is not None:
        budget_cap_tokens = fixed_cap_tokens
        budget_cap_source = "fixed"
    elif ewma_prefill_ms_per_token is None or run_query_index < warmup_queries:
        budget_cap_tokens = cap_bootstrap_tokens
        budget_cap_source = "bootstrap_warmup"
    else:
        budget_cap_tokens = int(round(prefill_target_ms / max(1e-6, ewma_prefill_ms_per_token)))
        budget_cap_source = "ewma_prefill"
    budget_cap_tokens = max(cap_min_tokens, min(cap_max_tokens, budget_cap_tokens))
    return {
        "prefill_target_ms": float(prefill_target_ms),
        "cap_bootstrap_tokens": int(cap_bootstrap_tokens),
        "cap_min_tokens": int(cap_min_tokens),
        "cap_max_tokens": int(cap_max_tokens),
        "fixed_cap_tokens": (int(fixed_cap_tokens) if fixed_cap_tokens is not None else None),
        "warmup_queries": int(warmup_queries),
        "ewma_prefill_ms_per_token": (
            float(ewma_prefill_ms_per_token) if ewma_prefill_ms_per_token is not None else None
        ),
        "run_query_index": int(run_query_index),
        "budget_cap_tokens": int(budget_cap_tokens),
        "budget_cap_source": str(budget_cap_source),
    }


def _resolve_agnostic_probe_task(cfg: dict[str, Any], cb: dict[str, Any]) -> tuple[str, CoverageGoal, int]:
    dataset_name = str(cfg.get("datasets", {}).get("qa", {}).get("name", "")).strip().lower()
    if dataset_name in {"hotpot_qa", "two_wiki_multihop"}:
        required_distinct_docs = max(1, int(cb.get("required_distinct_docs", 2)))
        return dataset_name, CoverageGoal.MULTI_DOCUMENT_EVIDENCE, required_distinct_docs
    raise ValueError(
        "context_budgeting.strategy=agnostic_acb_sc currently supports only "
        "datasets.qa.name in {hotpot_qa, two_wiki_multihop}"
    )


def _build_agnostic_probe_controller(cb: dict[str, Any]) -> ContextController:
    return ContextController(
        {
            "enabled": True,
            "stop_mode": str(cb.get("stop_mode", "coverage_locked_patience_v3")),
            "seed_min_items": int(cb.get("seed_min_items", 3)),
            "snippet_words": int(cb.get("snippet_words", 120)),
            "min_snippet_words": int(cb.get("min_snippet_words", 40)),
            "utility_weights": {
                "relevance": float(cb.get("utility_relevance_weight", 0.55)),
                "question_overlap": float(cb.get("utility_question_overlap_weight", 0.25)),
                "novelty": float(cb.get("utility_novelty_weight", 0.10)),
                "new_doc_bonus": float(cb.get("utility_new_doc_weight", 0.10)),
            },
            "patience": int(cb.get("patience", 2)),
            "single_evidence_extra_probe_candidates": int(cb.get("single_evidence_extra_probe_candidates", 0)),
            "multi_document_spare_probe_candidates": int(cb.get("multi_document_spare_probe_candidates", 1)),
            "multi_document_exact_probe_candidates": int(cb.get("multi_document_exact_probe_candidates", 2)),
            "marginal_snippet_enabled": bool(cb.get("marginal_snippet_enabled", True)),
            "marginal_snippet_ratio": float(cb.get("marginal_snippet_ratio", 0.60)),
            "marginal_snippet_min_words": int(cb.get("marginal_snippet_min_words", 40)),
            "marginal_snippet_max_words": int(cb.get("marginal_snippet_max_words", 72)),
        }
    )


def _to_agnostic_candidates(hits: list[RetrievedItem]) -> list[RetrievedCandidate]:
    candidates: list[RetrievedCandidate] = []
    for hit in hits:
        text = str(hit.text or "").strip()
        if not text:
            continue
        candidates.append(
            RetrievedCandidate(
                item_id=str(hit.item_id),
                doc_id=str(hit.doc_id),
                text=text,
                score=float(_safe_float(hit.score) or 0.0),
                source=str(hit.source or "legacy"),
            )
        )
    return candidates


def _adaptive_pack_contexts_agnostic_acb_sc(
    question: str,
    hits: list[RetrievedItem],
    retrieval_stages: dict[str, Any],
    cfg: dict[str, Any],
    margin_threshold: float,
    resolved: dict[str, Any],
    runtime_state: dict[str, Any] | None = None,
) -> tuple[list[str], float, dict[str, Any]]:
    t0 = time.perf_counter()
    retrieval_cfg = cfg.get("retrieval", {})
    cb = dict(cfg.get("context_budgeting", {}))
    runtime = dict(runtime_state or {})
    dataset_name, coverage_goal, required_distinct_docs = _resolve_agnostic_probe_task(cfg, cb)

    k_max_cfg = int(retrieval_cfg.get("top_k_final", len(hits)))
    k_max = max(1, min(len(hits), k_max_cfg, int(cb.get("max_chunks_hard_cap", resolved.get("max_chunks_hard_cap", 20)))))
    top_hits = hits[:k_max]
    budget = _resolve_probe_runtime_budget(cb, runtime)
    controller = _build_agnostic_probe_controller(cb)
    candidates = _to_agnostic_candidates(top_hits)
    result = controller.select(
        query=question,
        candidates=candidates,
        coverage_goal=coverage_goal,
        required_distinct_docs=required_distinct_docs,
        budget_cap_tokens=int(budget["budget_cap_tokens"]),
    )

    use_rerank = bool(cb.get("use_rerank_scores_if_available", resolved.get("use_rerank_scores_if_available", True)))
    margin_value, margin_source = _compute_margin(
        retrieval_stages,
        use_rerank_scores_if_available=use_rerank,
    )
    agreement_value = _safe_float(retrieval_stages.get("overlap_ratio", 0.0))
    agreement_value = float(agreement_value if agreement_value is not None else 0.0)
    candidate_trace = list(result.trace.get("candidate_trace", []))
    rejected_rows = [
        row
        for row in candidate_trace
        if not bool(row.get("selected", False)) and (bool(row.get("stop_decision", False)) or int(row.get("tokens_selected", 0)) <= 0)
    ]
    selected_rows = [row for row in candidate_trace if bool(row.get("selected", False))]
    selected_chunk_ids = [str(c.item_id) for c in result.selected_candidates]
    selected_doc_ids = _ordered_unique([str(c.doc_id) for c in result.selected_candidates])

    trace = {
        "enabled": True,
        "strategy": "agnostic_acb_sc",
        "stop_mode": str(result.trace.get("stop_mode", controller.stop_mode)),
        "dataset_name": str(dataset_name),
        "coverage_goal": str(result.trace.get("coverage_goal", coverage_goal.value)),
        "required_distinct_docs": int(required_distinct_docs),
        "k_eff": int(len(result.selected_candidates)),
        "context_budget_tokens": int(budget["budget_cap_tokens"]),
        "context_tokens_used": int(result.trace.get("context_tokens_used", 0)),
        "policy_branch": "dynamic",
        "margin_value": float(margin_value),
        "agreement_value": float(agreement_value),
        "margin_source": str(margin_source),
        "margin_threshold": float(margin_threshold),
        "agreement_threshold": 1.0,
        "keep_full_count": 0,
        "fallback_to_high": False,
        "fallback_reason": (
            "self_calibrated_stop"
            if rejected_rows and bool(rejected_rows[-1].get("stop_decision", False))
            else ("budget_cap" if int(result.trace.get("context_tokens_used", 0)) >= int(budget["budget_cap_tokens"]) else "")
        ),
        "context_chunk_ids_used": list(selected_chunk_ids),
        "context_doc_internal_ids_used": list(selected_doc_ids),
        "context_doc_ids_used": list(selected_doc_ids),
        "redundancy_avg_similarity": 0.0,
        "redundancy_max_similarity": 0.0,
        "aliases_used": dict(resolved.get("aliases_used", {})),
        "unknown_keys": list(resolved.get("unknown_keys", [])),
        "v2_resolved_params": {},
        "query_local_theta": float(result.trace.get("theta_q", 0.0)),
        "confidence_value": None,
        "budget_cap_source": str(budget["budget_cap_source"]),
        "budget_cap_tokens": int(budget["budget_cap_tokens"]),
        "fixed_cap_tokens": budget["fixed_cap_tokens"],
        "ewma_prefill_ms_per_token": budget["ewma_prefill_ms_per_token"],
        "prefill_target_ms": float(budget["prefill_target_ms"]),
        "warmup_queries": int(budget["warmup_queries"]),
        "seed_complete": bool(result.trace.get("coverage_unlocked", False)),
        "seed_min_unique_docs": int(required_distinct_docs),
        "seed_required_tokens": int(result.trace.get("dynamic_floor_tokens", 0)),
        "selected_candidates": list(selected_rows),
        "last_rejected_candidate": (dict(rejected_rows[-1]) if rejected_rows else {}),
        "agnostic_controller_trace": dict(result.trace),
        "utility_weights": dict(result.trace.get("utility_weights", {})),
    }
    return result.contexts, float((time.perf_counter() - t0) * 1000.0), trace


def _build_v2_contexts(
    question: str,
    hits: list[RetrievedItem],
    k_limit: int,
    budget_tokens: int,
    keep_full_count: int,
    snippet_from_rank: int,
    snippet_window_tokens: int,
    k_eff_floor: int,
    mmr_lambda: float,
    top_doc_saliency_tokens: int,
    saliency_entity_weight: float,
    dynamic_mmr_enabled: bool,
    dynamic_mmr_threshold: float,
    dynamic_mmr_boost: float,
    dynamic_mmr_cap: float,
) -> dict[str, Any]:
    k_limit = max(1, min(int(k_limit), len(hits)))
    budget_tokens = max(64, int(budget_tokens))
    keep_full_count = max(1, int(keep_full_count))
    snippet_from_rank = max(1, int(snippet_from_rank))
    snippet_window_tokens = max(24, int(snippet_window_tokens))
    k_eff_floor = max(1, min(int(k_eff_floor), k_limit))

    candidates: list[dict[str, Any]] = []
    for idx, hit in enumerate(hits[:k_limit]):
        full_text = str(hit.text or "").strip()
        if not full_text:
            continue
        full_tokens = _word_count(full_text)
        snippet_text = _snippet_window(full_text, question=question, window_tokens=snippet_window_tokens)
        snippet_tokens = _word_count(snippet_text)
        relevance = _safe_float(hit.score)
        candidates.append(
            {
                "index": idx,
                "rank": idx + 1,
                "hit": hit,
                "full_text": full_text,
                "full_tokens": full_tokens,
                "snippet_text": snippet_text,
                "snippet_tokens": snippet_tokens,
                "full_token_set": _text_token_set(full_text),
                "relevance": float(relevance if relevance is not None else 0.0),
            }
        )

    if not candidates:
        return {
            "contexts": [],
            "used_tokens": 0,
            "k_eff": 0,
            "context_chunk_ids_used": [],
            "context_doc_internal_ids_used": [],
            "context_doc_ids_used": [],
            "redundancy_avg_similarity": 0.0,
            "redundancy_max_similarity": 0.0,
        }

    rel_norm = _normalize_relevance([float(c["relevance"]) for c in candidates])
    for i, value in enumerate(rel_norm):
        candidates[i]["relevance_norm"] = float(value)

    selected: list[dict[str, Any]] = []
    selected_idx: set[int] = set()
    rejected_idx: set[int] = set()
    selected_token_sets: list[set[str]] = []
    used_tokens = 0
    mmr_lambda_max_used = float(mmr_lambda)

    # Mandatory top evidence blocks: saliency snippets with fixed budget per doc.
    for c in candidates:
        if c["rank"] > keep_full_count:
            break
        text = _saliency_snippet_with_title(
            text=str(c["full_text"]),
            question=question,
            doc_id=str(c["hit"].doc_id),
            budget_tokens=int(top_doc_saliency_tokens),
            entity_weight=float(saliency_entity_weight),
        )
        mode = "saliency"
        if not text:
            text = str(c["full_text"]).strip()
            mode = "full_fallback"
        if not text:
            continue
        tokens = _word_count(text)
        if tokens <= 0:
            continue
        selected.append(
            {
                "rank": int(c["rank"]),
                "hit": c["hit"],
                "text": text,
                "tokens": tokens,
                "mode": mode,
            }
        )
        selected_idx.add(int(c["index"]))
        selected_token_sets.append(_text_token_set(text))
        used_tokens += tokens

    while len(selected) < k_limit:
        remaining = [c for c in candidates if int(c["index"]) not in selected_idx and int(c["index"]) not in rejected_idx]
        if not remaining:
            break

        scored: list[tuple[float, int, str, dict[str, Any]]] = []
        for c in remaining:
            if selected_token_sets:
                max_sim = max(_jaccard(c["full_token_set"], sset) for sset in selected_token_sets)
            else:
                max_sim = 0.0
            lambda_eff = float(mmr_lambda)
            if dynamic_mmr_enabled and max_sim >= float(dynamic_mmr_threshold):
                lambda_eff = min(
                    float(dynamic_mmr_cap),
                    float(mmr_lambda) + float(dynamic_mmr_boost) * float(max_sim - dynamic_mmr_threshold),
                )
            mmr_lambda_max_used = max(mmr_lambda_max_used, float(lambda_eff))
            mmr_score = float(c["relevance_norm"]) - (float(lambda_eff) * float(max_sim))
            scored.append((mmr_score, -int(c["rank"]), str(c["hit"].item_id), c))
        scored.sort(reverse=True)
        chosen = scored[0][3]

        use_text = ""
        use_tokens = 0
        mode = ""

        full_text = str(chosen["full_text"])
        full_tokens = int(chosen["full_tokens"])
        snippet_text = str(chosen["snippet_text"])
        snippet_tokens = int(chosen["snippet_tokens"])

        if used_tokens + full_tokens <= budget_tokens:
            use_text = full_text
            use_tokens = full_tokens
            mode = "full"
        elif int(chosen["rank"]) >= snippet_from_rank and snippet_tokens > 0:
            if used_tokens + snippet_tokens <= budget_tokens:
                use_text = snippet_text
                use_tokens = snippet_tokens
                mode = "snippet"
            elif len(selected) < k_eff_floor:
                use_text = _leading_window(snippet_text, max(16, min(snippet_window_tokens, snippet_tokens)))
                use_tokens = _word_count(use_text)
                mode = "snippet_forced"
        elif len(selected) < k_eff_floor:
            fallback_window = max(16, min(snippet_window_tokens, full_tokens))
            use_text = _leading_window(full_text, fallback_window)
            use_tokens = _word_count(use_text)
            mode = "full_forced"

        if not use_text or use_tokens <= 0:
            rejected_idx.add(int(chosen["index"]))
            continue

        selected.append(
            {
                "rank": int(chosen["rank"]),
                "hit": chosen["hit"],
                "text": use_text,
                "tokens": int(use_tokens),
                "mode": mode,
            }
        )
        selected_idx.add(int(chosen["index"]))
        selected_token_sets.append(_text_token_set(use_text))
        used_tokens += int(use_tokens)

        if len(selected) >= k_eff_floor and used_tokens >= budget_tokens:
            break

    if len(selected) < k_eff_floor:
        for c in candidates:
            if int(c["index"]) in selected_idx:
                continue
            if len(selected) >= k_eff_floor:
                break
            fallback_text = ""
            fallback_tokens = 0
            fallback_mode = ""
            if int(c["rank"]) >= snippet_from_rank and int(c["snippet_tokens"]) > 0:
                fallback_text = str(c["snippet_text"])
                fallback_tokens = int(c["snippet_tokens"])
                fallback_mode = "snippet_floor"
            else:
                fallback_text = _leading_window(str(c["full_text"]), max(16, min(snippet_window_tokens, int(c["full_tokens"])) ))
                fallback_tokens = _word_count(fallback_text)
                fallback_mode = "full_floor"
            if not fallback_text or fallback_tokens <= 0:
                continue
            selected.append(
                {
                    "rank": int(c["rank"]),
                    "hit": c["hit"],
                    "text": fallback_text,
                    "tokens": int(fallback_tokens),
                    "mode": fallback_mode,
                }
            )
            selected_idx.add(int(c["index"]))
            selected_token_sets.append(_text_token_set(fallback_text))
            used_tokens += int(fallback_tokens)

    selected.sort(key=lambda x: (int(x["rank"]), str(x["hit"].item_id)))
    contexts = [str(x["text"]) for x in selected]
    chunk_ids = [str(x["hit"].item_id) for x in selected]
    doc_internal_ids = _ordered_unique([str(x["hit"].doc_id) for x in selected])
    doc_ids = list(doc_internal_ids)

    sims: list[float] = []
    final_sets = [_text_token_set(str(x["text"])) for x in selected]
    for i in range(len(final_sets)):
        for j in range(i + 1, len(final_sets)):
            sims.append(_jaccard(final_sets[i], final_sets[j]))

    return {
        "contexts": contexts,
        "used_tokens": int(used_tokens),
        "k_eff": int(len(contexts)),
        "context_chunk_ids_used": chunk_ids,
        "context_doc_internal_ids_used": doc_internal_ids,
        "context_doc_ids_used": doc_ids,
        "redundancy_avg_similarity": float(sum(sims) / len(sims)) if sims else 0.0,
        "redundancy_max_similarity": float(max(sims)) if sims else 0.0,
        "mmr_lambda_max_used": float(mmr_lambda_max_used),
    }


def _adaptive_pack_contexts_v2(
    question: str,
    hits: list[RetrievedItem],
    retrieval_stages: dict[str, Any],
    cfg: dict[str, Any],
    margin_threshold: float,
    resolved: dict[str, Any],
) -> tuple[list[str], float, dict[str, Any]]:
    t0 = time.perf_counter()
    retrieval_cfg = cfg.get("retrieval", {})

    k_max_cfg = int(retrieval_cfg.get("top_k_final", len(hits)))
    k_max = max(1, min(len(hits), k_max_cfg, int(resolved.get("max_chunks_hard_cap", 20))))

    margin_value, margin_source = _compute_margin(
        retrieval_stages,
        use_rerank_scores_if_available=bool(resolved.get("use_rerank_scores_if_available", True)),
    )
    agreement_value = _safe_float(retrieval_stages.get("overlap_ratio", 0.0))
    agreement_value = float(agreement_value if agreement_value is not None else 0.0)

    agreement_threshold = float(resolved.get("agreement_threshold", 0.35))
    low_margin_multiplier = float(resolved.get("low_margin_multiplier", 5.0))
    low_agreement_multiplier = float(resolved.get("low_agreement_multiplier", 1.0))
    low_condition = (
        margin_value >= float(margin_threshold) * low_margin_multiplier
        or agreement_value >= agreement_threshold * low_agreement_multiplier
    )
    medium_condition = (
        bool(resolved.get("medium_branch_enabled", True))
        and (margin_value >= float(margin_threshold) or agreement_value >= agreement_threshold)
    )

    if low_condition:
        branch = "low"
        k_limit = max(int(resolved.get("k_eff_floor", 5)), min(k_max, int(resolved.get("k_low", 5))))
        budget = int(resolved.get("budget_low_tokens", 600))
    elif medium_condition:
        branch = "medium"
        k_limit = max(int(resolved.get("k_eff_floor", 5)), min(k_max, int(resolved.get("medium_k_eff", 8))))
        budget = int(resolved.get("medium_budget_tokens", 900))
    else:
        branch = "high"
        k_limit = int(k_max)
        budget = int(resolved.get("budget_high_tokens", 1200))

    base_plan = _build_v2_contexts(
        question=question,
        hits=hits,
        k_limit=k_limit,
        budget_tokens=budget,
        keep_full_count=int(resolved.get("keep_full_count", 2)),
        snippet_from_rank=int(resolved.get("snippet_from_rank", 3)),
        snippet_window_tokens=int(resolved.get("snippet_window_tokens", 80)),
        k_eff_floor=int(resolved.get("k_eff_floor", 5)),
        mmr_lambda=float(resolved.get("mmr_lambda", 0.30)),
        top_doc_saliency_tokens=int(resolved.get("top_doc_saliency_tokens", 192)),
        saliency_entity_weight=float(resolved.get("saliency_entity_weight", 2.0)),
        dynamic_mmr_enabled=bool(resolved.get("dynamic_mmr_enabled", True)),
        dynamic_mmr_threshold=float(resolved.get("dynamic_mmr_threshold", 0.45)),
        dynamic_mmr_boost=float(resolved.get("dynamic_mmr_boost", 0.60)),
        dynamic_mmr_cap=float(resolved.get("dynamic_mmr_cap", 1.20)),
    )

    fallback_to_high = False
    fallback_reason = ""
    high_ref_tokens = int(base_plan.get("used_tokens", 0))
    chosen_branch = branch

    if branch != "high":
        high_plan = _build_v2_contexts(
            question=question,
            hits=hits,
            k_limit=k_max,
            budget_tokens=int(resolved.get("budget_high_tokens", 1200)),
            keep_full_count=int(resolved.get("keep_full_count", 2)),
            snippet_from_rank=int(resolved.get("snippet_from_rank", 3)),
            snippet_window_tokens=int(resolved.get("snippet_window_tokens", 80)),
            k_eff_floor=int(resolved.get("k_eff_floor", 5)),
            mmr_lambda=float(resolved.get("mmr_lambda", 0.30)),
            top_doc_saliency_tokens=int(resolved.get("top_doc_saliency_tokens", 192)),
            saliency_entity_weight=float(resolved.get("saliency_entity_weight", 2.0)),
            dynamic_mmr_enabled=bool(resolved.get("dynamic_mmr_enabled", True)),
            dynamic_mmr_threshold=float(resolved.get("dynamic_mmr_threshold", 0.45)),
            dynamic_mmr_boost=float(resolved.get("dynamic_mmr_boost", 0.60)),
            dynamic_mmr_cap=float(resolved.get("dynamic_mmr_cap", 1.20)),
        )
        high_ref_tokens = int(high_plan.get("used_tokens", 0))
        low_tokens = int(base_plan.get("used_tokens", 0))
        savings_ratio = float((high_ref_tokens - low_tokens) / max(1, high_ref_tokens))
        if savings_ratio < float(resolved.get("min_low_savings_ratio", 0.15)):
            fallback_to_high = True
            fallback_reason = "insufficient_low_savings"
            chosen_branch = "high"
            base_plan = high_plan

    trace = {
        "enabled": True,
        "strategy": "v2_evidence_first",
        "k_eff": int(base_plan.get("k_eff", 0)),
        "context_budget_tokens": int(budget),
        "context_tokens_used": int(base_plan.get("used_tokens", 0)),
        "policy_branch": chosen_branch,
        "initial_policy_branch": branch,
        "margin_value": float(margin_value),
        "agreement_value": float(agreement_value),
        "margin_source": margin_source,
        "margin_threshold": float(margin_threshold),
        "agreement_threshold": float(agreement_threshold),
        "keep_full_count": int(resolved.get("keep_full_count", 2)),
        "fallback_to_high": bool(fallback_to_high),
        "fallback_reason": fallback_reason,
        "high_branch_reference_tokens": int(high_ref_tokens),
        "context_chunk_ids_used": list(base_plan.get("context_chunk_ids_used", [])),
        "context_doc_internal_ids_used": list(base_plan.get("context_doc_internal_ids_used", [])),
        # These IDs are doc-level identifiers used for QA gold matching in this codebase.
        "context_doc_ids_used": list(base_plan.get("context_doc_ids_used", [])),
        "redundancy_avg_similarity": float(base_plan.get("redundancy_avg_similarity", 0.0)),
        "redundancy_max_similarity": float(base_plan.get("redundancy_max_similarity", 0.0)),
        "mmr_lambda_max_used": float(base_plan.get("mmr_lambda_max_used", resolved.get("mmr_lambda", 0.30))),
        "v2_resolved_params": {
            "keep_full_count": int(resolved.get("keep_full_count", 2)),
            "mmr_lambda": float(resolved.get("mmr_lambda", 0.30)),
            "min_low_savings_ratio": float(resolved.get("min_low_savings_ratio", 0.15)),
            "k_eff_floor": int(resolved.get("k_eff_floor", 5)),
            "snippet_from_rank": int(resolved.get("snippet_from_rank", 3)),
            "snippet_window_tokens": int(resolved.get("snippet_window_tokens", 80)),
            "low_margin_multiplier": float(resolved.get("low_margin_multiplier", 5.0)),
            "low_agreement_multiplier": float(resolved.get("low_agreement_multiplier", 1.0)),
            "medium_branch_enabled": bool(resolved.get("medium_branch_enabled", True)),
            "medium_budget_tokens": int(resolved.get("medium_budget_tokens", 900)),
            "medium_k_eff": int(resolved.get("medium_k_eff", 8)),
            "budget_low_tokens": int(resolved.get("budget_low_tokens", 600)),
            "budget_high_tokens": int(resolved.get("budget_high_tokens", 1200)),
            "max_chunks_hard_cap": int(resolved.get("max_chunks_hard_cap", 20)),
            "top_doc_saliency_tokens": int(resolved.get("top_doc_saliency_tokens", 192)),
            "saliency_entity_weight": float(resolved.get("saliency_entity_weight", 2.0)),
            "dynamic_mmr_enabled": bool(resolved.get("dynamic_mmr_enabled", True)),
            "dynamic_mmr_threshold": float(resolved.get("dynamic_mmr_threshold", 0.45)),
            "dynamic_mmr_boost": float(resolved.get("dynamic_mmr_boost", 0.60)),
            "dynamic_mmr_cap": float(resolved.get("dynamic_mmr_cap", 1.20)),
        },
        "aliases_used": dict(resolved.get("aliases_used", {})),
        "unknown_keys": list(resolved.get("unknown_keys", [])),
    }

    return list(base_plan.get("contexts", [])), float((time.perf_counter() - t0) * 1000.0), trace


def _adaptive_pack_contexts_incremental(
    question: str,
    hits: list[RetrievedItem],
    retrieval_stages: dict[str, Any],
    cfg: dict[str, Any],
    margin_threshold: float,
    resolved: dict[str, Any],
) -> tuple[list[str], float, dict[str, Any]]:
    t0 = time.perf_counter()
    retrieval_cfg = cfg.get("retrieval", {})
    cb = dict(cfg.get("context_budgeting", {}))

    k_max_cfg = int(retrieval_cfg.get("top_k_final", len(hits)))
    k_max = max(1, min(len(hits), k_max_cfg, int(cb.get("max_chunks_hard_cap", resolved.get("max_chunks_hard_cap", 20)))))
    use_rerank = bool(cb.get("use_rerank_scores_if_available", resolved.get("use_rerank_scores_if_available", True)))
    agreement_threshold = float(cb.get("agreement_threshold", resolved.get("agreement_threshold", 0.35)))

    margin_value, margin_source = _compute_margin(
        retrieval_stages,
        use_rerank_scores_if_available=use_rerank,
    )
    agreement_value = _safe_float(retrieval_stages.get("overlap_ratio", 0.0))
    agreement_value = float(agreement_value if agreement_value is not None else 0.0)

    keep_full_count = max(1, min(k_max, int(cb.get("keep_full_count", 2))))
    min_hits_floor = max(keep_full_count, min(k_max, int(cb.get("min_hits_floor", 5))))
    min_tokens_before_stop = max(64, int(cb.get("min_tokens_before_stop", cb.get("budget_low_tokens", 600))))
    budget_cap_tokens = max(min_tokens_before_stop, int(cb.get("budget_cap_tokens", cb.get("budget_high_tokens", 1200))))
    snippet_from_rank = max(1, int(cb.get("snippet_from_rank", 4)))
    prefer_full_until_rank = max(keep_full_count, int(cb.get("prefer_full_until_rank", max(keep_full_count, snippet_from_rank - 1))))
    snippet_window_tokens = max(24, int(cb.get("snippet_window_tokens", 80)))
    full_relevance_floor = max(0.0, min(1.0, float(cb.get("full_relevance_floor", 0.60))))

    utility_stop_threshold = max(0.0, min(1.5, float(cb.get("utility_stop_threshold", 0.42))))
    confidence_stop_weight = max(0.0, float(cb.get("confidence_stop_weight", 0.08)))
    relevance_weight = max(0.0, float(cb.get("utility_relevance_weight", 0.55)))
    overlap_weight = max(0.0, float(cb.get("utility_question_overlap_weight", 0.20)))
    novelty_weight = max(0.0, float(cb.get("utility_novelty_weight", 0.15)))
    new_doc_weight = max(0.0, float(cb.get("utility_new_doc_weight", 0.10)))
    stop_mode = str(cb.get("stop_mode", "triple_condition")).strip().lower() or "triple_condition"
    if stop_mode not in {"triple_condition", "utility_only"}:
        stop_mode = "triple_condition"
    stop_mode = str(cb.get("stop_mode", "triple_condition")).strip().lower() or "triple_condition"
    if stop_mode not in {"triple_condition", "utility_only"}:
        stop_mode = "triple_condition"

    top_hits = hits[:k_max]
    rel_norm = _normalize_relevance([float(_safe_float(h.score) or 0.0) for h in top_hits])
    q_tokens = _question_token_set(question)

    margin_ratio = float(margin_value / max(1e-9, float(margin_threshold))) if float(margin_threshold) > 0.0 else 0.0
    agreement_ratio = float(agreement_value / max(1e-9, agreement_threshold)) if agreement_threshold > 0.0 else 0.0
    confidence_value = max(margin_ratio, agreement_ratio)
    effective_stop_threshold = float(
        min(1.25, max(0.05, utility_stop_threshold + (confidence_stop_weight * (confidence_value - 1.0))))
    )

    packed_contexts: list[str] = []
    context_chunk_ids: list[str] = []
    context_doc_internal_ids: list[str] = []
    selected_token_sets: list[set[str]] = []
    seen_doc_ids: set[str] = set()
    used_tokens = 0
    stop_reason = ""

    for idx, hit in enumerate(top_hits):
        rank = idx + 1
        raw = str(hit.text or "").strip()
        if not raw:
            continue

        full_text = raw
        full_tokens = _word_count(full_text)
        snippet_text = _snippet_window(full_text, question=question, window_tokens=snippet_window_tokens)
        snippet_tokens = _word_count(snippet_text)
        rel_score = float(rel_norm[idx]) if idx < len(rel_norm) else 0.0

        candidate_text = full_text
        candidate_tokens = full_tokens
        candidate_mode = "full"
        if rank > prefer_full_until_rank and (rel_score < full_relevance_floor or used_tokens + full_tokens > budget_cap_tokens):
            if snippet_tokens > 0:
                candidate_text = snippet_text
                candidate_tokens = snippet_tokens
                candidate_mode = "snippet"

        candidate_token_set = _text_token_set(candidate_text)
        max_similarity = max((_jaccard(candidate_token_set, s) for s in selected_token_sets), default=0.0)
        q_overlap = float(len(candidate_token_set.intersection(q_tokens)) / max(1, len(q_tokens)))
        new_doc_bonus = 1.0 if str(hit.doc_id) not in seen_doc_ids else 0.0
        novelty = 1.0 - max_similarity
        utility = (
            (relevance_weight * rel_score)
            + (overlap_weight * q_overlap)
            + (novelty_weight * novelty)
            + (new_doc_weight * new_doc_bonus)
        )

        must_keep = (len(packed_contexts) < min_hits_floor) or (used_tokens < min_tokens_before_stop) or (rank <= keep_full_count)
        if (not must_keep) and utility < effective_stop_threshold:
            stop_reason = "low_marginal_utility"
            break

        remaining = budget_cap_tokens - used_tokens
        if remaining <= 0:
            stop_reason = "budget_cap"
            break
        if candidate_tokens > remaining:
            candidate_text = _leading_window(candidate_text, remaining)
            candidate_tokens = _word_count(candidate_text)
            candidate_mode = f"{candidate_mode}_trimmed"
        if candidate_tokens <= 0:
            stop_reason = "budget_cap"
            break

        packed_contexts.append(candidate_text)
        context_chunk_ids.append(str(hit.item_id))
        context_doc_internal_ids.append(str(hit.doc_id))
        used_tokens += candidate_tokens
        selected_token_sets.append(_text_token_set(candidate_text))
        seen_doc_ids.add(str(hit.doc_id))

        if used_tokens >= budget_cap_tokens:
            stop_reason = "budget_cap"
            break

    if not packed_contexts and top_hits:
        fallback = _leading_window(str(top_hits[0].text or "").strip(), min(budget_cap_tokens, snippet_window_tokens))
        if fallback:
            packed_contexts.append(fallback)
            context_chunk_ids.append(str(top_hits[0].item_id))
            context_doc_internal_ids.append(str(top_hits[0].doc_id))
            used_tokens = _word_count(fallback)
            stop_reason = "fallback_first_hit"

    trace = {
        "enabled": True,
        "strategy": "incremental_stop",
        "k_eff": int(len(packed_contexts)),
        "context_budget_tokens": int(budget_cap_tokens),
        "context_tokens_used": int(used_tokens),
        "policy_branch": "dynamic",
        "margin_value": float(margin_value),
        "agreement_value": float(agreement_value),
        "margin_source": margin_source,
        "margin_threshold": float(margin_threshold),
        "agreement_threshold": float(agreement_threshold),
        "keep_full_count": int(keep_full_count),
        "fallback_to_high": False,
        "fallback_reason": stop_reason,
        "context_chunk_ids_used": list(context_chunk_ids),
        "context_doc_internal_ids_used": _ordered_unique(context_doc_internal_ids),
        "context_doc_ids_used": _ordered_unique(context_doc_internal_ids),
        "redundancy_avg_similarity": 0.0,
        "redundancy_max_similarity": 0.0,
        "aliases_used": dict(resolved.get("aliases_used", {})),
        "unknown_keys": list(resolved.get("unknown_keys", [])),
        "v2_resolved_params": {},
        "utility_stop_threshold": float(utility_stop_threshold),
        "effective_stop_threshold": float(effective_stop_threshold),
        "confidence_value": float(confidence_value),
    }
    return packed_contexts, float((time.perf_counter() - t0) * 1000.0), trace


def _adaptive_pack_contexts_incremental_sc(
    question: str,
    hits: list[RetrievedItem],
    retrieval_stages: dict[str, Any],
    cfg: dict[str, Any],
    margin_threshold: float,
    resolved: dict[str, Any],
    runtime_state: dict[str, Any] | None = None,
) -> tuple[list[str], float, dict[str, Any]]:
    t0 = time.perf_counter()
    retrieval_cfg = cfg.get("retrieval", {})
    cb = dict(cfg.get("context_budgeting", {}))
    runtime = dict(runtime_state or {})

    k_max_cfg = int(retrieval_cfg.get("top_k_final", len(hits)))
    k_max = max(1, min(len(hits), k_max_cfg, int(cb.get("max_chunks_hard_cap", resolved.get("max_chunks_hard_cap", 20)))))
    use_rerank = bool(cb.get("use_rerank_scores_if_available", resolved.get("use_rerank_scores_if_available", True)))

    keep_full_count = max(1, min(k_max, int(cb.get("keep_full_count", 2))))
    seed_min_unique_docs = max(1, int(cb.get("seed_min_unique_docs", 3)))
    seed_token_fraction = max(0.10, min(0.95, float(cb.get("seed_token_fraction", 0.35))))
    snippet_from_rank = max(1, int(cb.get("snippet_from_rank", 4)))
    prefer_full_until_rank = max(keep_full_count, int(cb.get("prefer_full_until_rank", max(keep_full_count, snippet_from_rank - 1))))
    snippet_window_tokens = max(24, int(cb.get("snippet_window_tokens", 80)))
    full_relevance_floor = max(0.0, min(1.0, float(cb.get("full_relevance_floor", 0.60))))
    relevance_weight = max(0.0, float(cb.get("utility_relevance_weight", 0.55)))
    overlap_weight = max(0.0, float(cb.get("utility_question_overlap_weight", 0.20)))
    novelty_weight = max(0.0, float(cb.get("utility_novelty_weight", 0.15)))
    new_doc_weight = max(0.0, float(cb.get("utility_new_doc_weight", 0.10)))

    stop_mode = str(cb.get("stop_mode", "triple_condition")).strip().lower() or "triple_condition"
    if stop_mode not in {"triple_condition", "utility_only"}:
        stop_mode = "triple_condition"

    prefill_target_ms = max(1000.0, float(cb.get("prefill_target_ms", 18000.0)))
    cap_bootstrap_tokens = max(128, int(cb.get("cap_bootstrap_tokens", 1024)))
    cap_min_tokens = max(64, int(cb.get("cap_min_tokens", 768)))
    cap_max_tokens = max(cap_min_tokens, int(cb.get("cap_max_tokens", 1536)))
    fixed_cap_tokens_raw = cb.get("fixed_cap_tokens", None)
    fixed_cap_tokens = None if fixed_cap_tokens_raw is None else max(64, int(fixed_cap_tokens_raw))
    warmup_queries = max(0, int(cb.get("warmup_queries", 8)))
    ewma_prefill_ms_per_token = _safe_float(runtime.get("ewma_prefill_ms_per_token"))
    run_query_index = max(0, int(runtime.get("query_index", 0)))
    if fixed_cap_tokens is not None:
        budget_cap_tokens = fixed_cap_tokens
        budget_cap_source = "fixed"
    elif ewma_prefill_ms_per_token is None or run_query_index < warmup_queries:
        budget_cap_tokens = cap_bootstrap_tokens
        budget_cap_source = "bootstrap_warmup"
    else:
        budget_cap_tokens = int(round(prefill_target_ms / max(1e-6, ewma_prefill_ms_per_token)))
        budget_cap_source = "ewma_prefill"
    budget_cap_tokens = max(cap_min_tokens, min(cap_max_tokens, budget_cap_tokens))

    top_hits = hits[:k_max]
    rel_norm = _normalize_relevance([float(_safe_float(h.score) or 0.0) for h in top_hits])
    q_tokens = _question_token_set(question)

    score_values, margin_source = _score_list(retrieval_stages, use_rerank_scores_if_available=use_rerank)
    score_values = score_values[: max(2, min(6, len(score_values)))]
    gaps = [max(0.0, score_values[i] - score_values[i + 1]) for i in range(max(0, len(score_values) - 1))]
    query_local_tau = max(1e-6, _quantile(gaps, 0.9)) if gaps else max(1e-6, float(margin_threshold))
    margin_value = float(gaps[0]) if gaps else 0.0
    agreement_value = _safe_float(retrieval_stages.get("overlap_ratio", 0.0))
    agreement_value = float(agreement_value if agreement_value is not None else 0.0)
    margin_conf = min(2.0, float(margin_value / max(1e-6, query_local_tau)))
    agreement_conf = min(1.0, max(0.0, agreement_value))
    confidence_value = float((0.6 * margin_conf) + (0.4 * agreement_conf))

    packed_contexts: list[str] = []
    context_chunk_ids: list[str] = []
    context_doc_internal_ids: list[str] = []
    selected_token_sets: list[set[str]] = []
    seen_doc_ids: set[str] = set()
    used_tokens = 0
    stop_reason = ""
    seed_complete = False
    query_local_theta = 0.0
    seed_utilities: list[float] = []
    seed_novelties: list[float] = []
    seed_q_overlaps: list[float] = []
    selected_trace: list[dict[str, Any]] = []
    last_rejected_trace: dict[str, Any] | None = None

    for idx, hit in enumerate(top_hits):
        rank = idx + 1
        raw = str(hit.text or "").strip()
        if not raw:
            continue

        full_text = raw
        full_tokens = _word_count(full_text)
        snippet_text = _snippet_window(full_text, question=question, window_tokens=snippet_window_tokens)
        snippet_tokens = _word_count(snippet_text)
        rel_score = float(rel_norm[idx]) if idx < len(rel_norm) else 0.0

        candidate_text = full_text
        candidate_tokens = full_tokens
        candidate_mode = "full"
        if rank > prefer_full_until_rank and (rel_score < full_relevance_floor or used_tokens + full_tokens > budget_cap_tokens):
            if snippet_tokens > 0:
                candidate_text = snippet_text
                candidate_tokens = snippet_tokens
                candidate_mode = "snippet"

        remaining = budget_cap_tokens - used_tokens
        if remaining <= 0:
            stop_reason = "budget_cap"
            break
        if candidate_tokens > remaining:
            candidate_text = _leading_window(candidate_text, remaining)
            candidate_tokens = _word_count(candidate_text)
            candidate_mode = f"{candidate_mode}_trimmed"
        if candidate_tokens <= 0:
            stop_reason = "budget_cap"
            break

        candidate_token_set = _text_token_set(candidate_text)
        max_similarity = max((_jaccard(candidate_token_set, s) for s in selected_token_sets), default=0.0)
        q_overlap = float(len(candidate_token_set.intersection(q_tokens)) / max(1, len(q_tokens)))
        new_doc_bonus = 1.0 if str(hit.doc_id) not in seen_doc_ids else 0.0
        novelty = 1.0 - max_similarity
        relevance_component = relevance_weight * rel_score
        overlap_component = overlap_weight * q_overlap
        novelty_component = novelty_weight * novelty
        new_doc_component = new_doc_weight * new_doc_bonus
        utility = relevance_component + overlap_component + novelty_component + new_doc_component

        candidate_info = {
            "rank": int(rank),
            "doc_id": str(hit.doc_id),
            "chunk_id": str(hit.item_id),
            "mode": candidate_mode,
            "tokens": int(candidate_tokens),
            "relevance_component": float(relevance_component),
            "question_overlap_component": float(overlap_component),
            "novelty_component": float(novelty_component),
            "new_doc_component": float(new_doc_component),
            "utility": float(utility),
            "novelty": float(novelty),
            "question_overlap": float(q_overlap),
            "new_doc_bonus": float(new_doc_bonus),
        }

        seed_required_tokens = max(1, int(round(seed_token_fraction * budget_cap_tokens)))
        current_unique_docs = len(seen_doc_ids)
        must_keep = (len(packed_contexts) < keep_full_count) or (not seed_complete)
        should_stop = False

        seed_phase = not seed_complete
        if seed_complete:
            novelty_gate = float(novelty) < _median(seed_novelties)
            overlap_gate = str(hit.doc_id) in seen_doc_ids or float(q_overlap) < _median(seed_q_overlaps)
            if stop_mode == "utility_only":
                should_stop = float(utility) < float(query_local_theta)
            else:
                should_stop = (
                    float(utility) < float(query_local_theta)
                    and novelty_gate
                    and overlap_gate
                )
            candidate_info["stop_rule"] = stop_mode
            candidate_info["stop_decision"] = bool(should_stop)
            candidate_info["theta_q"] = float(query_local_theta)
            candidate_info["novelty_gate"] = bool(novelty_gate)
            candidate_info["overlap_gate"] = bool(overlap_gate)
        else:
            candidate_info["stop_rule"] = "seed_build"
            candidate_info["stop_decision"] = False
            candidate_info["theta_q"] = None
            candidate_info["novelty_gate"] = None
            candidate_info["overlap_gate"] = None

        if (not must_keep) and should_stop:
            stop_reason = "self_calibrated_stop"
            last_rejected_trace = dict(candidate_info)
            break

        packed_contexts.append(candidate_text)
        context_chunk_ids.append(str(hit.item_id))
        context_doc_internal_ids.append(str(hit.doc_id))
        used_tokens += candidate_tokens
        selected_token_sets.append(candidate_token_set)
        seen_doc_ids.add(str(hit.doc_id))
        candidate_info["decision"] = "selected"
        candidate_info["seed_phase"] = bool(seed_phase)
        selected_trace.append(dict(candidate_info))
        if seed_phase:
            seed_utilities.append(float(utility))
            seed_novelties.append(float(novelty))
            seed_q_overlaps.append(float(q_overlap))

        if not seed_complete:
            seed_complete = (len(seen_doc_ids) >= seed_min_unique_docs) and (used_tokens >= seed_required_tokens)
            if seed_complete:
                query_local_theta = float(min(1.25, max(0.05, _median(seed_utilities) - _mad(seed_utilities))))

        if used_tokens >= budget_cap_tokens:
            stop_reason = "budget_cap"
            break

    if not packed_contexts and top_hits:
        fallback = _leading_window(str(top_hits[0].text or "").strip(), min(budget_cap_tokens, snippet_window_tokens))
        if fallback:
            packed_contexts.append(fallback)
            context_chunk_ids.append(str(top_hits[0].item_id))
            context_doc_internal_ids.append(str(top_hits[0].doc_id))
            used_tokens = _word_count(fallback)
            stop_reason = "fallback_first_hit"

    if not seed_complete and packed_contexts:
        stop_reason = stop_reason or "seed_incomplete_used_available"
        query_local_theta = float(min(1.25, max(0.05, _median(seed_utilities) - _mad(seed_utilities))))

    selected_mean_trace = {
        "relevance_component": float(_median([float(x.get("relevance_component", 0.0)) for x in selected_trace])),
        "question_overlap_component": float(_median([float(x.get("question_overlap_component", 0.0)) for x in selected_trace])),
        "novelty_component": float(_median([float(x.get("novelty_component", 0.0)) for x in selected_trace])),
        "new_doc_component": float(_median([float(x.get("new_doc_component", 0.0)) for x in selected_trace])),
        "utility": float(_median([float(x.get("utility", 0.0)) for x in selected_trace])),
    }

    trace = {
        "enabled": True,
        "strategy": "incremental_sc",
        "stop_mode": str(stop_mode),
        "k_eff": int(len(packed_contexts)),
        "context_budget_tokens": int(budget_cap_tokens),
        "context_tokens_used": int(used_tokens),
        "policy_branch": "dynamic",
        "margin_value": float(margin_value),
        "agreement_value": float(agreement_value),
        "margin_source": margin_source,
        "margin_threshold": float(query_local_tau),
        "agreement_threshold": 1.0,
        "keep_full_count": int(keep_full_count),
        "fallback_to_high": False,
        "fallback_reason": stop_reason,
        "context_chunk_ids_used": list(context_chunk_ids),
        "context_doc_internal_ids_used": _ordered_unique(context_doc_internal_ids),
        "context_doc_ids_used": _ordered_unique(context_doc_internal_ids),
        "redundancy_avg_similarity": 0.0,
        "redundancy_max_similarity": 0.0,
        "aliases_used": dict(resolved.get("aliases_used", {})),
        "unknown_keys": list(resolved.get("unknown_keys", [])),
        "v2_resolved_params": {},
        "query_local_tau": float(query_local_tau),
        "query_local_theta": float(query_local_theta),
        "confidence_value": float(confidence_value),
        "budget_cap_source": str(budget_cap_source),
        "budget_cap_tokens": int(budget_cap_tokens),
        "fixed_cap_tokens": (int(fixed_cap_tokens) if fixed_cap_tokens is not None else None),
        "ewma_prefill_ms_per_token": (
            float(ewma_prefill_ms_per_token) if ewma_prefill_ms_per_token is not None else None
        ),
        "prefill_target_ms": float(prefill_target_ms),
        "warmup_queries": int(warmup_queries),
        "seed_complete": bool(seed_complete),
        "seed_min_unique_docs": int(seed_min_unique_docs),
        "seed_required_tokens": int(max(1, int(round(seed_token_fraction * budget_cap_tokens)))),
        "selected_utility_means": dict(selected_mean_trace),
        "selected_candidates": list(selected_trace),
        "last_rejected_candidate": dict(last_rejected_trace or {}),
    }
    return packed_contexts, float((time.perf_counter() - t0) * 1000.0), trace


def _adaptive_pack_contexts_v1(
    question: str,
    hits: list[RetrievedItem],
    retrieval_stages: dict[str, Any],
    cfg: dict[str, Any],
    margin_threshold: float,
    resolved: dict[str, Any],
) -> tuple[list[str], float, dict[str, Any]]:
    t0 = time.perf_counter()
    retrieval_cfg = cfg.get("retrieval", {})
    llm_cfg = cfg.get("llm", {})

    k_max_cfg = int(retrieval_cfg.get("top_k_final", len(hits)))
    k_max = max(1, min(len(hits), k_max_cfg))
    k_low = max(1, int(resolved.get("k_low", 5)))
    budget_low = max(64, int(resolved.get("budget_low_tokens", 600)))
    budget_high = max(budget_low, int(resolved.get("budget_high_tokens", 1200)))
    agreement_threshold = float(resolved.get("agreement_threshold", 0.35))
    use_rerank = bool(resolved.get("use_rerank_scores_if_available", True))
    max_chunks_hard_cap = max(1, int(resolved.get("max_chunks_hard_cap", 20)))
    snippet_window_tokens = max(24, int(resolved.get("snippet_window_tokens", 80)))
    packing_mode_low = str(resolved.get("packing_mode_low", "full_or_light"))
    packing_mode_high = str(resolved.get("packing_mode_high", "snippet"))
    llm_context_packing = bool(llm_cfg.get("context_packing", False))
    llm_pack_words = max(16, int(llm_cfg.get("context_pack_words", 80)))

    margin_value, margin_source = _compute_margin(
        retrieval_stages,
        use_rerank_scores_if_available=use_rerank,
    )
    agreement_value = _safe_float(retrieval_stages.get("overlap_ratio", 0.0))
    agreement_value = float(agreement_value if agreement_value is not None else 0.0)

    low_branch = (margin_value >= float(margin_threshold)) or (agreement_value >= agreement_threshold)
    if low_branch:
        branch = "low"
        k_eff = min(k_low, k_max)
        budget = budget_low
        packing = packing_mode_low
    else:
        branch = "high"
        k_eff = k_max
        budget = budget_high
        packing = packing_mode_high

    k_eff = min(k_eff, max_chunks_hard_cap, len(hits))
    selected = hits[:k_eff]

    packed_contexts: list[str] = []
    context_chunk_ids: list[str] = []
    context_doc_internal_ids: list[str] = []
    used_tokens = 0
    for hit in selected:
        raw = hit.text.strip()
        if not raw:
            continue
        if packing == "snippet":
            candidate = _snippet_window(raw, question=question, window_tokens=snippet_window_tokens)
        else:
            if llm_context_packing:
                candidate = _leading_window(raw, llm_pack_words)
            else:
                candidate = raw

        if not candidate:
            continue
        c_tokens = _word_count(candidate)
        remaining = budget - used_tokens
        if remaining <= 0:
            break
        if c_tokens > remaining:
            candidate = _leading_window(candidate, remaining)
            c_tokens = _word_count(candidate)
        if c_tokens <= 0:
            continue
        packed_contexts.append(candidate)
        context_chunk_ids.append(str(hit.item_id))
        context_doc_internal_ids.append(str(hit.doc_id))
        used_tokens += c_tokens
        if used_tokens >= budget:
            break

    trace = {
        "enabled": True,
        "strategy": "v1",
        "k_eff": int(k_eff),
        "context_budget_tokens": int(budget),
        "context_tokens_used": int(used_tokens),
        "policy_branch": branch,
        "margin_value": float(margin_value),
        "agreement_value": float(agreement_value),
        "margin_source": margin_source,
        "margin_threshold": float(margin_threshold),
        "agreement_threshold": float(agreement_threshold),
        "keep_full_count": 0,
        "fallback_to_high": False,
        "fallback_reason": "",
        "context_chunk_ids_used": list(context_chunk_ids),
        "context_doc_internal_ids_used": _ordered_unique(context_doc_internal_ids),
        "context_doc_ids_used": _ordered_unique(context_doc_internal_ids),
        "redundancy_avg_similarity": 0.0,
        "redundancy_max_similarity": 0.0,
        "aliases_used": dict(resolved.get("aliases_used", {})),
        "unknown_keys": list(resolved.get("unknown_keys", [])),
        "v2_resolved_params": {},
    }
    return packed_contexts, float((time.perf_counter() - t0) * 1000.0), trace


def adaptive_pack_contexts(
    question: str,
    hits: list[RetrievedItem],
    retrieval_stages: dict[str, Any],
    cfg: dict[str, Any],
    margin_threshold: float,
    resolved_context_budget: dict[str, Any] | None = None,
    runtime_state: dict[str, Any] | None = None,
) -> tuple[list[str], float, dict[str, Any]]:
    t0 = time.perf_counter()
    cb = dict(cfg.get("context_budgeting", {}))
    resolved = dict(resolved_context_budget or resolve_context_budgeting_config(cb))
    enabled = bool(resolved.get("enabled", False))
    strategy = str(resolved.get("strategy", "v1")).lower()

    if not enabled:
        return [h.text for h in hits], 0.0, {
            "enabled": False,
            "strategy": strategy,
            "k_eff": int(min(len(hits), int(cfg.get("retrieval", {}).get("top_k_final", len(hits))))),
            "context_budget_tokens": 0,
            "context_tokens_used": 0,
            "policy_branch": "disabled",
            "margin_value": None,
            "agreement_value": None,
            "margin_source": "none",
            "margin_threshold": float(margin_threshold),
            "agreement_threshold": float(resolved.get("agreement_threshold", 0.35)),
            "keep_full_count": 0,
            "fallback_to_high": False,
            "fallback_reason": "",
            "context_chunk_ids_used": [str(h.item_id) for h in hits],
            "context_doc_internal_ids_used": _ordered_unique([str(h.doc_id) for h in hits]),
            "context_doc_ids_used": _ordered_unique([str(h.doc_id) for h in hits]),
            "redundancy_avg_similarity": 0.0,
            "redundancy_max_similarity": 0.0,
            "aliases_used": dict(resolved.get("aliases_used", {})),
            "unknown_keys": list(resolved.get("unknown_keys", [])),
            "v2_resolved_params": {},
            "pack_elapsed_ms": float((time.perf_counter() - t0) * 1000.0),
        }

    if strategy == "v2_evidence_first":
        contexts, elapsed_ms, trace = _adaptive_pack_contexts_v2(
            question=question,
            hits=hits,
            retrieval_stages=retrieval_stages,
            cfg=cfg,
            margin_threshold=margin_threshold,
            resolved=resolved,
        )
    elif strategy == "incremental_stop":
        contexts, elapsed_ms, trace = _adaptive_pack_contexts_incremental(
            question=question,
            hits=hits,
            retrieval_stages=retrieval_stages,
            cfg=cfg,
            margin_threshold=margin_threshold,
            resolved=resolved,
        )
    elif strategy == "incremental_sc":
        contexts, elapsed_ms, trace = _adaptive_pack_contexts_incremental_sc(
            question=question,
            hits=hits,
            retrieval_stages=retrieval_stages,
            cfg=cfg,
            margin_threshold=margin_threshold,
            resolved=resolved,
            runtime_state=runtime_state,
        )
    elif strategy == "agnostic_acb_sc":
        contexts, elapsed_ms, trace = _adaptive_pack_contexts_agnostic_acb_sc(
            question=question,
            hits=hits,
            retrieval_stages=retrieval_stages,
            cfg=cfg,
            margin_threshold=margin_threshold,
            resolved=resolved,
            runtime_state=runtime_state,
        )
    else:
        contexts, elapsed_ms, trace = _adaptive_pack_contexts_v1(
            question=question,
            hits=hits,
            retrieval_stages=retrieval_stages,
            cfg=cfg,
            margin_threshold=margin_threshold,
            resolved=resolved,
        )

    trace.setdefault("strategy", strategy)
    trace.setdefault("aliases_used", dict(resolved.get("aliases_used", {})))
    trace.setdefault("unknown_keys", list(resolved.get("unknown_keys", [])))
    trace.setdefault("v2_resolved_params", {})
    trace["pack_elapsed_ms"] = float(elapsed_ms)
    return contexts, float(elapsed_ms), trace
