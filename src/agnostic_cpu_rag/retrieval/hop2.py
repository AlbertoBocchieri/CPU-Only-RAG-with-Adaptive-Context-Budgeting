from __future__ import annotations

import re
from typing import Any

from ..records import RetrievedCandidate
from ..utils import clamp, head_words, minmax_normalize

_BRIDGE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
_STOP_TERMS = {
    # Pronouns: these are generic referential words, not bridge entities.
    "It",
    "He",
    "She",
    "They",
    "This",
    "That",
    "These",
    "Those",
    # Articles: title-cased at sentence boundaries but never useful bridge terms.
    "A",
    "An",
    "The",
}


def extract_bridge_terms(query: str, candidates: list[RetrievedCandidate], max_terms: int = 2) -> list[str]:
    query_lower = str(query).lower()
    scored: dict[str, float] = {}
    for rank, cand in enumerate(candidates, start=1):
        title = str(cand.title or "").strip()
        if len(title.split()) >= 2 and title.lower() not in query_lower:
            scored[title] = max(scored.get(title, 0.0), 2.0 / rank)
        first_sentence = head_words(cand.text, 32)
        for match in _BRIDGE_RE.findall(first_sentence):
            term = " ".join(match.split()).strip()
            if not term or term in _STOP_TERMS or term.lower() in query_lower:
                continue
            if len(term.split()) == 1 and len(term) < 5:
                continue
            scored[term] = max(scored.get(term, 0.0), 1.0 / rank)
    ranked = sorted(scored.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    return [term for term, _ in ranked[: max(0, int(max_terms))]]


def should_activate_hop2(
    *,
    fusion_trace: dict[str, Any],
    retrieval_budget_ms: float,
    estimated_extra_cost_ms: float,
    activation_conf_threshold: float,
    agreement_threshold: float,
) -> bool:
    combined_conf = max(float(fusion_trace.get("conf_dense", 0.0)), float(fusion_trace.get("conf_lexical", 0.0)))
    agreement = float(fusion_trace.get("agreement", 0.0))
    budget_ok = float(estimated_extra_cost_ms) <= float(retrieval_budget_ms)
    low_conf = combined_conf < float(activation_conf_threshold)
    low_agreement = agreement < float(agreement_threshold)
    return bool(budget_ok and (low_conf or low_agreement))


def build_hop2_queries(query: str, bridge_terms: list[str], max_queries: int = 2) -> list[str]:
    out: list[str] = []
    for term in bridge_terms[: max(0, int(max_queries))]:
        expanded = f"{str(query).strip()} {str(term).strip()}".strip()
        if expanded and expanded not in out:
            out.append(expanded)
    return out


def conservative_novel_slot_merge(
    *,
    base_ranked: list[RetrievedCandidate],
    hop2_ranked: list[RetrievedCandidate],
    top_k_final: int,
    reserved_novel_slots: int = 2,
    min_normalized_score: float = 0.35,
) -> tuple[list[RetrievedCandidate], dict[str, Any]]:
    top_k_final = max(1, int(top_k_final))
    reserved_novel_slots = max(0, min(int(reserved_novel_slots), top_k_final))
    keep_base = max(1, top_k_final - reserved_novel_slots)
    kept = list(base_ranked[:keep_base])
    kept_item_ids = {c.item_id for c in kept}
    kept_doc_ids = {c.doc_id for c in kept}
    hop2_scores = {c.item_id: float(c.score) for c in hop2_ranked}
    hop2_norm = minmax_normalize(hop2_scores)
    injected: list[RetrievedCandidate] = []
    for cand in hop2_ranked:
        if cand.item_id in kept_item_ids or cand.doc_id in kept_doc_ids:
            continue
        if float(hop2_norm.get(cand.item_id, 0.0)) < float(min_normalized_score):
            continue
        injected.append(cand)
        kept_item_ids.add(cand.item_id)
        kept_doc_ids.add(cand.doc_id)
        if len(injected) >= reserved_novel_slots:
            break
    merged = kept + injected
    if len(merged) < top_k_final:
        for cand in base_ranked[keep_base:]:
            if cand.item_id in kept_item_ids:
                continue
            merged.append(cand)
            kept_item_ids.add(cand.item_id)
            if len(merged) >= top_k_final:
                break
    return merged[:top_k_final], {
        "reserved_novel_slots": int(reserved_novel_slots),
        "min_normalized_score": float(min_normalized_score),
        "injected_item_ids": [c.item_id for c in injected],
        "injected_doc_ids": [c.doc_id for c in injected],
    }
