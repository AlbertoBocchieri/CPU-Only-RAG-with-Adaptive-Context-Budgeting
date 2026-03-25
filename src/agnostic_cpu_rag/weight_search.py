from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .adapters.tasks import make_task_adapter
from .context_controller import ContextController
from .evaluation import evaluate_query, summarize_query_records
from .records import CoverageGoal, GoldReference, RetrievedCandidate

WEIGHT_KEYS = ("relevance", "question_overlap", "novelty", "new_doc_bonus")
LEGACY_INHERITED_WEIGHTS = {
    "relevance": 0.55,
    "question_overlap": 0.20,
    "novelty": 0.15,
    "new_doc_bonus": 0.10,
}
BALANCED_MULTI_HOP_SEEDS = [
    {"relevance": 0.45, "question_overlap": 0.25, "novelty": 0.15, "new_doc_bonus": 0.15},
    {"relevance": 0.40, "question_overlap": 0.30, "novelty": 0.15, "new_doc_bonus": 0.15},
    {"relevance": 0.40, "question_overlap": 0.20, "novelty": 0.25, "new_doc_bonus": 0.15},
    {"relevance": 0.40, "question_overlap": 0.20, "novelty": 0.10, "new_doc_bonus": 0.30},
    {"relevance": 0.50, "question_overlap": 0.20, "novelty": 0.10, "new_doc_bonus": 0.20},
    {"relevance": 0.35, "question_overlap": 0.25, "novelty": 0.15, "new_doc_bonus": 0.25},
]
BALANCED_OPEN_QA_SEEDS = [
    {"relevance": 0.50, "question_overlap": 0.20, "novelty": 0.10, "new_doc_bonus": 0.20},
    {"relevance": 0.45, "question_overlap": 0.25, "novelty": 0.10, "new_doc_bonus": 0.20},
    {"relevance": 0.45, "question_overlap": 0.20, "novelty": 0.15, "new_doc_bonus": 0.20},
    {"relevance": 0.40, "question_overlap": 0.25, "novelty": 0.10, "new_doc_bonus": 0.25},
    {"relevance": 0.55, "question_overlap": 0.15, "novelty": 0.10, "new_doc_bonus": 0.20},
]
ANCHOR_SEARCH_RADIUS = 0.10
ANCHOR_SEARCH_STEP = 0.05
PILOT75_ACCEPTABLE_F1_DROP = -0.01
MULTI_HOP_BALANCED_BOUNDS = {
    "relevance": (0.30, 0.60),
    "question_overlap": (0.10, 0.30),
    "novelty": (0.10, 0.30),
    "new_doc_bonus": (0.10, 0.30),
}
OPEN_QA_BALANCED_BOUNDS = {
    "relevance": (0.35, 0.60),
    "question_overlap": (0.10, 0.30),
    "novelty": (0.05, 0.20),
    "new_doc_bonus": (0.10, 0.30),
}
MULTI_HOP_LOCAL_BOUNDS = {
    "relevance": (0.45, 0.65),
    "question_overlap": (0.10, 0.30),
    "novelty": (0.10, 0.20),
    "new_doc_bonus": (0.05, 0.20),
}
OPEN_QA_LOCAL_BOUNDS = {
    "relevance": (0.45, 0.65),
    "question_overlap": (0.10, 0.30),
    "novelty": (0.10, 0.20),
    "new_doc_bonus": (0.05, 0.20),
}


class WeightSearchError(RuntimeError):
    pass


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def canonicalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    out = {key: float(weights.get(key, 0.0)) for key in WEIGHT_KEYS}
    total = sum(out.values())
    if total <= 0.0:
        raise WeightSearchError("Utility weights must sum to a positive value")
    return {key: float(value / total) for key, value in out.items()}


def weight_signature(weights: Mapping[str, float]) -> str:
    norm = canonicalize_weights(weights)
    return "__".join(f"{key}={norm[key]:.2f}" for key in WEIGHT_KEYS)


def generate_weight_grid(
    *,
    step: float,
    include: Sequence[Mapping[str, float]] | None = None,
) -> list[dict[str, float]]:
    if step <= 0.0:
        raise WeightSearchError("step must be positive")
    resolution = round(1.0 / step)
    if abs((resolution * step) - 1.0) > 1e-9:
        raise WeightSearchError("step must evenly divide 1.0")

    out: list[dict[str, float]] = []
    seen: set[str] = set()
    for a in range(resolution + 1):
        for b in range(resolution + 1 - a):
            for c in range(resolution + 1 - a - b):
                d = resolution - a - b - c
                weights = {
                    "relevance": a * step,
                    "question_overlap": b * step,
                    "novelty": c * step,
                    "new_doc_bonus": d * step,
                }
                sig = weight_signature(weights)
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(canonicalize_weights(weights))
    for extra in include or []:
        sig = weight_signature(extra)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(canonicalize_weights(extra))
    return sorted(out, key=weight_signature)


def generate_local_refine_grid(
    *,
    coarse_survivors: Sequence[Mapping[str, float]],
    step: float = 0.05,
    radius: float = 0.10,
    include: Sequence[Mapping[str, float]] | None = None,
) -> list[dict[str, float]]:
    survivors = [canonicalize_weights(weights) for weights in coarse_survivors]
    if not survivors:
        return []
    base_grid = generate_weight_grid(step=step, include=include)
    out: list[dict[str, float]] = []
    seen: set[str] = set()
    for candidate in base_grid:
        for anchor in survivors:
            if all(abs(candidate[key] - anchor[key]) <= (radius + 1e-9) for key in WEIGHT_KEYS):
                sig = weight_signature(candidate)
                if sig not in seen:
                    seen.add(sig)
                    out.append(candidate)
                break
    return sorted(out, key=weight_signature)


def l1_distance_to_anchor(weights: Mapping[str, float], anchor: Mapping[str, float] | None = None) -> float:
    norm = canonicalize_weights(weights)
    anchor_norm = canonicalize_weights(anchor or LEGACY_INHERITED_WEIGHTS)
    return float(sum(abs(norm[key] - anchor_norm[key]) for key in WEIGHT_KEYS))


def within_anchor_radius(
    weights: Mapping[str, float],
    *,
    anchor: Mapping[str, float] | None = None,
    radius: float = ANCHOR_SEARCH_RADIUS,
) -> bool:
    norm = canonicalize_weights(weights)
    anchor_norm = canonicalize_weights(anchor or LEGACY_INHERITED_WEIGHTS)
    return all(abs(norm[key] - anchor_norm[key]) <= radius + 1e-9 for key in WEIGHT_KEYS)


def relevance_is_dominant(weights: Mapping[str, float]) -> bool:
    norm = canonicalize_weights(weights)
    return all(norm["relevance"] + 1e-9 >= norm[key] for key in WEIGHT_KEYS if key != "relevance")


def within_weight_bounds(weights: Mapping[str, float], bounds: Mapping[str, tuple[float, float]]) -> bool:
    norm = canonicalize_weights(weights)
    return all(bounds[key][0] <= norm[key] <= bounds[key][1] for key in WEIGHT_KEYS if key in bounds)


def filter_weight_grid(
    weights_grid: Sequence[Mapping[str, float]],
    *,
    bounds: Mapping[str, tuple[float, float]],
    include: Sequence[Mapping[str, float]] | None = None,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    seen: set[str] = set()
    for weights in list(weights_grid) + list(include or []):
        if not within_weight_bounds(weights, bounds):
            continue
        norm = canonicalize_weights(weights)
        sig = weight_signature(norm)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(norm)
    return sorted(out, key=weight_signature)


def build_anchor_local_grid(
    *,
    step: float = ANCHOR_SEARCH_STEP,
    anchor: Mapping[str, float] | None = None,
    radius: float = ANCHOR_SEARCH_RADIUS,
    bounds: Mapping[str, tuple[float, float]],
    include: Sequence[Mapping[str, float]] | None = None,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    seen: set[str] = set()
    for weights in list(generate_weight_grid(step=step, include=include)) + list(include or []):
        if not within_anchor_radius(weights, anchor=anchor, radius=radius):
            continue
        if not within_weight_bounds(weights, bounds):
            continue
        if not relevance_is_dominant(weights):
            continue
        norm = canonicalize_weights(weights)
        sig = weight_signature(norm)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(norm)
    return sorted(out, key=lambda item: (l1_distance_to_anchor(item, anchor), weight_signature(item)))


def question_prefix_bucket(question: str) -> str:
    token = (str(question).strip().lower().split() or ["other"])[0]
    if token in {"who", "what", "when", "where", "which", "how"}:
        return token
    return "other"


def answer_length_bucket(answers: Sequence[str]) -> str:
    lengths = [len(str(answer).strip().split()) for answer in answers if str(answer).strip()]
    if not lengths:
        return "1"
    min_len = min(lengths)
    if min_len <= 1:
        return "1"
    if min_len <= 3:
        return "2_3"
    return "4_plus"


def serialize_candidate(candidate: RetrievedCandidate) -> dict[str, Any]:
    return {
        "item_id": candidate.item_id,
        "doc_id": candidate.doc_id,
        "text": candidate.text,
        "score": candidate.score,
        "source": candidate.source,
        "title": candidate.title,
        "metadata": dict(candidate.metadata),
    }


def deserialize_candidate(payload: Mapping[str, Any]) -> RetrievedCandidate:
    return RetrievedCandidate(
        item_id=str(payload.get("item_id", "")),
        doc_id=str(payload.get("doc_id", "")),
        text=str(payload.get("text", "")),
        score=float(payload.get("score", 0.0)),
        source=str(payload.get("source", "")),
        title=str(payload.get("title", "")),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def serialize_gold(gold: GoldReference) -> dict[str, Any]:
    return {
        "qid": gold.qid,
        "answers": list(gold.answers),
        "relevant_doc_ids": sorted(gold.relevant_doc_ids),
        "metadata": dict(gold.metadata),
    }


def deserialize_gold(payload: Mapping[str, Any]) -> GoldReference:
    return GoldReference(
        qid=str(payload.get("qid", "")),
        answers=[str(value) for value in payload.get("answers", [])],
        relevant_doc_ids={str(value) for value in payload.get("relevant_doc_ids", [])},
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _coverage_goal(value: str) -> CoverageGoal:
    key = str(value).strip().lower()
    for goal in CoverageGoal:
        if goal.value == key:
            return goal
    raise WeightSearchError(f"Unsupported coverage goal in cache: {value}")


def evaluate_controller_cache(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_family: str,
    controller_cfg: Mapping[str, Any],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    task_adapter = make_task_adapter(task_family)
    merged_cfg = dict(controller_cfg)
    merged_cfg["utility_weights"] = canonicalize_weights(weights)
    controller = ContextController(merged_cfg)

    records: list[dict[str, Any]] = []
    for row in rows:
        query = str(row.get("query", ""))
        candidates = [deserialize_candidate(item) for item in row.get("candidates", [])]
        gold = deserialize_gold(dict(row.get("gold", {}) or {}))
        coverage_goal = _coverage_goal(str(row.get("coverage_goal", task_adapter.coverage_goal.value)))
        required_distinct_docs = int(row.get("required_distinct_docs", task_adapter.required_distinct_docs()))
        budget_cap_tokens = int(row.get("budget_cap_tokens", 0))
        result = controller.select(
            query=query,
            candidates=candidates,
            coverage_goal=coverage_goal,
            required_distinct_docs=required_distinct_docs,
            budget_cap_tokens=budget_cap_tokens,
        )
        selected_doc_ids = list(result.trace.get("context_doc_ids_used", []))
        record = {
            "qid": str(row.get("qid", gold.qid)),
            "query": query,
            "prediction": None,
            "retrieved_doc_ids": [candidate.doc_id for candidate in candidates],
            "selected_doc_ids": selected_doc_ids,
            "context_controller": dict(result.trace),
            "latency_ms": {
                "context_tokens_used": float(result.trace.get("context_tokens_used", 0.0)),
                "budget_cap_tokens": float(result.trace.get("budget_cap_tokens", budget_cap_tokens)),
            },
            "runtime": {
                "budget_cap_tokens": float(result.trace.get("budget_cap_tokens", budget_cap_tokens)),
                "budget_cap_source": str(row.get("budget_cap_source", "controller_cache")),
            },
        }
        record["metrics"] = evaluate_query(
            task_adapter=task_adapter,
            gold=gold,
            prediction=None,
            selected_doc_ids=selected_doc_ids,
        )
        records.append(record)

    summary = summarize_query_records(records)
    return {
        "summary": summary,
        "records": records,
        "task_family": task_family,
        "weights": canonicalize_weights(weights),
        "weight_signature": weight_signature(weights),
    }


def metric_mean(summary: Mapping[str, Any], key: str) -> float:
    if key in {"em", "f1", "relevant_doc_recall", "coverage_goal_met", "pair_in_context"}:
        return float(dict(summary.get("metrics_mean", {}) or {}).get(key, 0.0))
    if key in {"context_tokens_used", "budget_cap_tokens"}:
        stats = dict(dict(summary.get("latency_summary_ms", {}) or {}).get(key, {}) or {})
        return float(stats.get("mean", 0.0))
    stats = dict(dict(summary.get("context_controller_summary", {}) or {}).get(key, {}) or {})
    return float(stats.get("mean", 0.0))


def passes_multi_hop_gate(summary: Mapping[str, Any], dataset_name: str) -> bool:
    if dataset_name == "hotpot_qa":
        return (
            metric_mean(summary, "pair_in_context") >= 0.80
            and metric_mean(summary, "relevant_doc_recall") >= 0.89
            and metric_mean(summary, "context_tokens_used") <= 390.0
        )
    if dataset_name == "two_wiki_multihop":
        return (
            metric_mean(summary, "pair_in_context") >= 0.68
            and metric_mean(summary, "relevant_doc_recall") >= 0.73
            and metric_mean(summary, "context_tokens_used") <= 440.0
        )
    raise WeightSearchError(f"Unsupported multi-hop dataset gate: {dataset_name}")


def passes_multi_hop_relative_gate(summary: Mapping[str, Any], baseline_summary: Mapping[str, Any]) -> bool:
    return (
        metric_mean(summary, "pair_in_context") >= (metric_mean(baseline_summary, "pair_in_context") - 0.01)
        and metric_mean(summary, "relevant_doc_recall") >= (metric_mean(baseline_summary, "relevant_doc_recall") - 0.01)
        and metric_mean(summary, "context_tokens_used") <= (metric_mean(baseline_summary, "context_tokens_used") + 15.0)
        and metric_mean(summary, "selected_count") <= (metric_mean(baseline_summary, "selected_count") + 0.25)
    )


def passes_open_qa_gate(summary: Mapping[str, Any]) -> bool:
    return (
        metric_mean(summary, "coverage_goal_met") >= 0.95
        and metric_mean(summary, "selected_count") <= 4.0
        and metric_mean(summary, "context_tokens_used") <= 350.0
    )


def passes_open_qa_relative_gate(summary: Mapping[str, Any], baseline_summary: Mapping[str, Any]) -> bool:
    return (
        metric_mean(summary, "coverage_goal_met") >= metric_mean(baseline_summary, "coverage_goal_met")
        and metric_mean(summary, "selected_count") <= (metric_mean(baseline_summary, "selected_count") + 0.25)
        and metric_mean(summary, "context_tokens_used") <= (metric_mean(baseline_summary, "context_tokens_used") + 25.0)
    )


def multi_hop_rank_key(summary: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        metric_mean(summary, "context_tokens_used"),
        -metric_mean(summary, "pair_in_context"),
        -metric_mean(summary, "relevant_doc_recall"),
        -metric_mean(summary, "coverage_goal_met"),
    )


def open_qa_rank_key(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        metric_mean(summary, "context_tokens_used"),
        metric_mean(summary, "selected_count"),
        -metric_mean(summary, "coverage_goal_met"),
    )


def pooled_multi_hop_rank_key(hotpot_summary: Mapping[str, Any], twowiki_summary: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        mean([
            metric_mean(hotpot_summary, "context_tokens_used"),
            metric_mean(twowiki_summary, "context_tokens_used"),
        ]),
        -mean([
            metric_mean(hotpot_summary, "pair_in_context"),
            metric_mean(twowiki_summary, "pair_in_context"),
        ]),
        -mean([
            metric_mean(hotpot_summary, "relevant_doc_recall"),
            metric_mean(twowiki_summary, "relevant_doc_recall"),
        ]),
    )


def anchor_distance_rank_key(
    weights: Mapping[str, float],
    *,
    anchor: Mapping[str, float] | None = None,
) -> tuple[float, str]:
    return (l1_distance_to_anchor(weights, anchor), weight_signature(weights))


def landscape_is_flat(hotpot_summaries: Sequence[Mapping[str, Any]], twowiki_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def _ranges(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        pair_values = [metric_mean(summary, "pair_in_context") for summary in rows]
        context_values = [metric_mean(summary, "context_tokens_used") for summary in rows]
        return {
            "pair_in_context_range": max(pair_values) - min(pair_values),
            "context_mean_range": max(context_values) - min(context_values),
        }

    hotpot = _ranges(hotpot_summaries)
    twowiki = _ranges(twowiki_summaries)
    is_flat = (
        hotpot["pair_in_context_range"] < 0.02
        and hotpot["context_mean_range"] < 10.0
        and twowiki["pair_in_context_range"] < 0.02
        and twowiki["context_mean_range"] < 10.0
    )
    return {
        "hotpot": hotpot,
        "two_wiki_multihop": twowiki,
        "flat": bool(is_flat),
    }


def true_lodo_survivors(
    coarse_scores: Mapping[str, Mapping[str, Any]],
    *,
    selection_top_k: int = 20,
    selection_fallback_top_k: int = 40,
) -> dict[str, Any]:
    def _select(top_k: int) -> dict[str, Any]:
        hotpot_candidates = [
            sig for sig, row in coarse_scores.items() if passes_multi_hop_gate(row["hotpot_qa"], "hotpot_qa")
        ]
        twowiki_candidates = [
            sig
            for sig, row in coarse_scores.items()
            if passes_multi_hop_gate(row["two_wiki_multihop"], "two_wiki_multihop")
        ]
        hotpot_ranked = sorted(hotpot_candidates, key=lambda sig: multi_hop_rank_key(coarse_scores[sig]["hotpot_qa"]))
        twowiki_ranked = sorted(
            twowiki_candidates,
            key=lambda sig: multi_hop_rank_key(coarse_scores[sig]["two_wiki_multihop"]),
        )
        fold1_selected = hotpot_ranked[:top_k]
        fold2_selected = twowiki_ranked[:top_k]
        fold1_valid = [
            sig
            for sig in fold1_selected
            if passes_multi_hop_gate(coarse_scores[sig]["two_wiki_multihop"], "two_wiki_multihop")
        ]
        fold2_valid = [
            sig for sig in fold2_selected if passes_multi_hop_gate(coarse_scores[sig]["hotpot_qa"], "hotpot_qa")
        ]
        survivors = sorted(set(fold1_valid).intersection(fold2_valid))
        return {
            "top_k": int(top_k),
            "fold1_selected": fold1_selected,
            "fold1_valid": fold1_valid,
            "fold2_selected": fold2_selected,
            "fold2_valid": fold2_valid,
            "survivors": survivors,
        }

    primary = _select(selection_top_k)
    if primary["survivors"]:
        primary["method"] = "true_lodo"
        return primary
    fallback = _select(selection_fallback_top_k)
    if fallback["survivors"]:
        fallback["method"] = "true_lodo_top40"
        return fallback

    joint = sorted(
        [
            sig
            for sig, row in coarse_scores.items()
            if passes_multi_hop_gate(row["hotpot_qa"], "hotpot_qa")
            and passes_multi_hop_gate(row["two_wiki_multihop"], "two_wiki_multihop")
        ],
        key=lambda sig: pooled_multi_hop_rank_key(
            coarse_scores[sig]["hotpot_qa"],
            coarse_scores[sig]["two_wiki_multihop"],
        ),
    )
    return {
        "top_k": int(selection_fallback_top_k),
        "fold1_selected": primary["fold1_selected"],
        "fold1_valid": primary["fold1_valid"],
        "fold2_selected": primary["fold2_selected"],
        "fold2_valid": primary["fold2_valid"],
        "survivors": joint,
        "method": "joint_gate_fallback",
    }


def true_lodo_survivors_relative(
    coarse_scores: Mapping[str, Mapping[str, Any]],
    *,
    hotpot_baseline: Mapping[str, Any],
    twowiki_baseline: Mapping[str, Any],
    selection_top_k: int = 20,
    selection_fallback_top_k: int = 40,
) -> dict[str, Any]:
    def _select(top_k: int) -> dict[str, Any]:
        hotpot_candidates = [
            sig
            for sig, row in coarse_scores.items()
            if passes_multi_hop_relative_gate(row["hotpot_qa"], hotpot_baseline)
        ]
        twowiki_candidates = [
            sig
            for sig, row in coarse_scores.items()
            if passes_multi_hop_relative_gate(row["two_wiki_multihop"], twowiki_baseline)
        ]
        hotpot_ranked = sorted(hotpot_candidates, key=lambda sig: multi_hop_rank_key(coarse_scores[sig]["hotpot_qa"]))
        twowiki_ranked = sorted(
            twowiki_candidates,
            key=lambda sig: multi_hop_rank_key(coarse_scores[sig]["two_wiki_multihop"]),
        )
        fold1_selected = hotpot_ranked[:top_k]
        fold2_selected = twowiki_ranked[:top_k]
        fold1_valid = [
            sig
            for sig in fold1_selected
            if passes_multi_hop_relative_gate(coarse_scores[sig]["two_wiki_multihop"], twowiki_baseline)
        ]
        fold2_valid = [
            sig
            for sig in fold2_selected
            if passes_multi_hop_relative_gate(coarse_scores[sig]["hotpot_qa"], hotpot_baseline)
        ]
        survivors = sorted(set(fold1_valid).intersection(fold2_valid))
        return {
            "top_k": int(top_k),
            "fold1_selected": fold1_selected,
            "fold1_valid": fold1_valid,
            "fold2_selected": fold2_selected,
            "fold2_valid": fold2_valid,
            "survivors": survivors,
        }

    primary = _select(selection_top_k)
    if primary["survivors"]:
        primary["method"] = "true_lodo_relative_to_inherited"
        return primary
    fallback = _select(selection_fallback_top_k)
    if fallback["survivors"]:
        fallback["method"] = "true_lodo_relative_to_inherited_top40"
        return fallback

    joint = sorted(
        [
            sig
            for sig, row in coarse_scores.items()
            if passes_multi_hop_relative_gate(row["hotpot_qa"], hotpot_baseline)
            and passes_multi_hop_relative_gate(row["two_wiki_multihop"], twowiki_baseline)
        ],
        key=lambda sig: pooled_multi_hop_rank_key(
            coarse_scores[sig]["hotpot_qa"],
            coarse_scores[sig]["two_wiki_multihop"],
        ),
    )
    return {
        "top_k": int(selection_fallback_top_k),
        "fold1_selected": primary["fold1_selected"],
        "fold1_valid": primary["fold1_valid"],
        "fold2_selected": primary["fold2_selected"],
        "fold2_valid": primary["fold2_valid"],
        "survivors": joint,
        "method": "joint_relative_to_inherited_fallback",
    }


def true_lodo_relative_filter(
    scores: Mapping[str, Mapping[str, Any]],
    *,
    hotpot_baseline: Mapping[str, Any],
    twowiki_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    fold1_candidates = [
        sig for sig, row in scores.items() if passes_multi_hop_relative_gate(row["hotpot_qa"], hotpot_baseline)
    ]
    fold2_candidates = [
        sig for sig, row in scores.items() if passes_multi_hop_relative_gate(row["two_wiki_multihop"], twowiki_baseline)
    ]
    fold1_valid = [
        sig for sig in fold1_candidates if passes_multi_hop_relative_gate(scores[sig]["two_wiki_multihop"], twowiki_baseline)
    ]
    fold2_valid = [
        sig for sig in fold2_candidates if passes_multi_hop_relative_gate(scores[sig]["hotpot_qa"], hotpot_baseline)
    ]
    survivors = sorted(set(fold1_valid).intersection(fold2_valid))
    return {
        "method": "true_lodo_relative_filter",
        "fold1_candidates": fold1_candidates,
        "fold2_candidates": fold2_candidates,
        "fold1_valid": fold1_valid,
        "fold2_valid": fold2_valid,
        "survivors": survivors,
    }
