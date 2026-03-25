from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from rag_cpu.metrics import evaluate_retrieval, summarize_list

from .adapters.base import TaskAdapter
from .records import DatasetBundle, GoldReference


def build_qrels(bundle: DatasetBundle) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for qid, gold in bundle.gold_references.items():
        qrels[qid] = {doc_id: 1 for doc_id in gold.relevant_doc_ids}
    return qrels


def evaluate_query(
    *,
    task_adapter: TaskAdapter,
    gold: GoldReference,
    prediction: str | None,
    selected_doc_ids: list[str],
) -> dict[str, float | int | None]:
    if prediction is None:
        answer_metrics = {"em": None, "f1": None}
    else:
        answer_metrics = task_adapter.evaluate_answer(str(prediction), gold)
    context_metrics = task_adapter.evaluate_context(selected_doc_ids, gold)
    return {**answer_metrics, **context_metrics}


def summarize_query_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    scalars: dict[str, list[float]] = defaultdict(list)
    latency_fields = [
        "t_retrieval_total_ms",
        "t_rerank_total_ms",
        "t_llm_total_ms",
        "t_prefill_ms",
        "t_decode_total_ms",
        "ttft_ms",
        "t_total_ms",
        "context_tokens_used",
        "budget_cap_tokens",
    ]
    for row in records:
        metrics = row.get("metrics", {})
        for key in ("em", "f1", "relevant_doc_recall", "coverage_goal_met", "pair_in_context"):
            value = metrics.get(key)
            if value is not None:
                scalars[key].append(float(value))
        lat = row.get("latency_ms", {})
        ctx = row.get("context_controller", {})
        for key in latency_fields:
            if key in lat and lat[key] is not None:
                scalars[key].append(float(lat[key]))
        if ctx.get("context_tokens_used") is not None:
            scalars["context_tokens_used"].append(float(ctx["context_tokens_used"]))
        if ctx.get("budget_cap_tokens") is not None:
            scalars["budget_cap_tokens"].append(float(ctx["budget_cap_tokens"]))
        for key in (
            "selected_count",
            "selected_doc_count",
            "seed_item_count",
            "seed_distinct_doc_count",
            "seed_token_total",
            "seed_median_tokens",
            "theta_q",
        ):
            value = ctx.get(key)
            if value is not None:
                scalars[key].append(float(value))
        runtime = row.get("runtime", {})
        for key in ("ewma_prefill_ms_per_token", "ewma_decode_ms_per_token", "ewma_embed_ms_per_doc"):
            value = runtime.get(key)
            if value is not None:
                scalars[key].append(float(value))

    def summarize_scalar_list(values: list[float]) -> dict[str, float]:
        stats = dict(summarize_list(values))
        stats["std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        return stats

    out: dict[str, Any] = {
        "num_queries": len(records),
        "metrics_mean": {},
        "latency_summary_ms": {},
        "context_controller_summary": {},
        "runtime_summary": {},
    }
    for key in ("em", "f1", "relevant_doc_recall", "coverage_goal_met", "pair_in_context"):
        vals = scalars.get(key, [])
        if vals:
            out["metrics_mean"][key] = float(mean(vals))
    for key in latency_fields:
        vals = scalars.get(key, [])
        if vals:
            out["latency_summary_ms"][key] = summarize_scalar_list(vals)
    for key in (
        "selected_count",
        "selected_doc_count",
        "seed_item_count",
        "seed_distinct_doc_count",
        "seed_token_total",
        "seed_median_tokens",
        "theta_q",
    ):
        vals = scalars.get(key, [])
        if vals:
            out["context_controller_summary"][key] = summarize_scalar_list(vals)
    for key in ("ewma_prefill_ms_per_token", "ewma_decode_ms_per_token", "ewma_embed_ms_per_doc"):
        vals = scalars.get(key, [])
        if vals:
            out["runtime_summary"][key] = summarize_scalar_list(vals)
    return out


def evaluate_retrieval_run(
    *,
    rankings: dict[str, list[str]],
    bundle: DatasetBundle,
    ks: list[int],
) -> dict[str, Any]:
    agg, per_query = evaluate_retrieval(rankings, build_qrels(bundle), ks)
    return {"aggregate": agg, "per_query": per_query}
