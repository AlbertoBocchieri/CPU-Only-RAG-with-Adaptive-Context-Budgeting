from __future__ import annotations

from typing import Any

from rag_cpu.retrievers import RetrievedItem

from ..utils import clamp, entropy_concentration, minmax_normalize


def _scores_from_hits(hits: list[RetrievedItem], top_m: int = 5) -> list[float]:
    return [float(hit.score) for hit in hits[: max(1, int(top_m))]]


def _top_gap_norm(hits: list[RetrievedItem], top_m: int = 5) -> float:
    if not hits:
        return 0.0
    values = _scores_from_hits(hits, top_m=top_m)
    score_map = {str(i): float(v) for i, v in enumerate(values)}
    normalized = [float(v) for _, v in sorted(minmax_normalize(score_map).items())]
    if not normalized:
        return 0.0
    if len(normalized) == 1:
        return float(normalized[0])
    return float(max(0.0, normalized[0] - normalized[1]))


def _top_ids(hits: list[RetrievedItem], top_m: int = 5) -> set[str]:
    return {hit.item_id for hit in hits[: max(1, int(top_m))]}


def compute_dynamic_alpha(
    *,
    dense_hits: list[RetrievedItem],
    lexical_hits: list[RetrievedItem],
    low_conf_threshold: float = 0.2,
    top_m: int = 5,
) -> tuple[float, dict[str, Any]]:
    dense_gap = _top_gap_norm(dense_hits, top_m=top_m)
    lexical_gap = _top_gap_norm(lexical_hits, top_m=top_m)
    dense_concentration = entropy_concentration(_scores_from_hits(dense_hits, top_m=top_m))
    lexical_concentration = entropy_concentration(_scores_from_hits(lexical_hits, top_m=top_m))
    dense_top = _top_ids(dense_hits, top_m=top_m)
    lexical_top = _top_ids(lexical_hits, top_m=top_m)
    denom = max(1, min(len(dense_top), len(lexical_top)))
    agreement = float(len(dense_top & lexical_top) / denom) if dense_top and lexical_top else 0.0

    conf_dense = clamp((0.4 * dense_gap) + (0.4 * dense_concentration) + (0.2 * agreement), 0.0, 1.0)
    conf_lexical = clamp((0.4 * lexical_gap) + (0.4 * lexical_concentration) + (0.2 * agreement), 0.0, 1.0)

    source = "dynamic"
    if conf_dense < float(low_conf_threshold) and conf_lexical < float(low_conf_threshold):
        alpha_q = 0.5
        source = "fallback_equal_weight"
    else:
        alpha_q = float(conf_dense / max(1e-9, conf_dense + conf_lexical))

    trace = {
        "alpha_q": float(alpha_q),
        "alpha_source": source,
        "conf_dense": float(conf_dense),
        "conf_lexical": float(conf_lexical),
        "agreement": float(agreement),
        "dense_gap": float(dense_gap),
        "lexical_gap": float(lexical_gap),
        "dense_concentration": float(dense_concentration),
        "lexical_concentration": float(lexical_concentration),
        "low_conf_threshold": float(low_conf_threshold),
    }
    return float(alpha_q), trace


def fuse_weighted_sum(
    *,
    dense_hits: list[RetrievedItem],
    lexical_hits: list[RetrievedItem],
    alpha: float,
) -> dict[str, float]:
    dense_scores = {hit.item_id: float(hit.score) for hit in dense_hits}
    lexical_scores = {hit.item_id: float(hit.score) for hit in lexical_hits}
    dense_norm = minmax_normalize(dense_scores)
    lexical_norm = minmax_normalize(lexical_scores)
    all_ids = set(dense_norm) | set(lexical_norm)
    return {
        item_id: float((float(alpha) * dense_norm.get(item_id, 0.0)) + ((1.0 - float(alpha)) * lexical_norm.get(item_id, 0.0)))
        for item_id in all_ids
    }
