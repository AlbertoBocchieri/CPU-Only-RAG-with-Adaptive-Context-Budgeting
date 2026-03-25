#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return float(ordered[low])
    frac = pos - low
    return float((ordered[low] * (1.0 - frac)) + (ordered[high] * frac))


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "p50": float(median(values)),
        "p90": float(quantile(values, 0.90)),
        "p95": float(quantile(values, 0.95)),
        "max": float(max(values)),
    }


def nested_get(node: dict[str, Any], *parts: str) -> Any:
    value: Any = node
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def detect_schema(row: dict[str, Any]) -> str:
    if "context_controller" in row or "runtime" in row:
        return "agnostic"
    if "context_budgeting" in row or "answer_metrics_per_query" in row:
        return "legacy"
    return "unknown"


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    agg: dict[str, list[float]] = {
        "em": [],
        "f1": [],
        "pair_in_context": [],
        "support_doc_in_context_at_2": [],
        "relevant_doc_recall": [],
    }
    for row in rows:
        schema = detect_schema(row)
        if schema == "agnostic":
            metrics = row.get("metrics", {})
            mappings = {
                "em": metrics.get("em"),
                "f1": metrics.get("f1"),
                "pair_in_context": metrics.get("pair_in_context"),
                "relevant_doc_recall": metrics.get("relevant_doc_recall"),
                "support_doc_in_context_at_2": metrics.get("support_doc_in_context_at_2"),
            }
        else:
            answer = row.get("answer_metrics_per_query", {})
            post = row.get("post_context", {})
            support = answer.get("support_doc_metrics", {}) if isinstance(answer.get("support_doc_metrics"), dict) else {}
            mappings = {
                "em": answer.get("em"),
                "f1": answer.get("f1"),
                "pair_in_context": post.get("pair_in_context_at_k"),
                "support_doc_in_context_at_2": post.get("support_doc_in_context_at_2"),
                "relevant_doc_recall": support.get("support_doc_recall@10"),
            }
        for key, value in mappings.items():
            if value is not None:
                agg[key].append(float(value))
    return {key: float(mean(values)) for key, values in agg.items() if values}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize controller/runtime extras from a per_query.jsonl run.")
    parser.add_argument("per_query_jsonl")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = load_rows(Path(args.per_query_jsonl))
    summary_payload: dict[str, Any] = {}
    if args.summary_json:
        summary_payload = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))

    selected_count: list[float] = []
    selected_doc_count: list[float] = []
    seed_size: list[float] = []
    theta_values: list[float] = []
    coverage_goal_met: list[float] = []
    context_tokens: list[float] = []
    budget_caps: list[float] = []
    budget_sources: list[str] = []
    ewma_prefill: list[float] = []
    t_prefill: list[float] = []
    prefill_targets: list[float] = []
    post_warmup_abs_errors: list[float] = []
    post_warmup_overshoots = 0
    post_warmup_count = 0
    warmup_queries = 0
    transition_query_index = None
    last_bootstrap_query_index = None

    for idx, row in enumerate(rows):
        schema = detect_schema(row)
        if schema == "agnostic":
            controller = row.get("context_controller", {})
            runtime = row.get("runtime", {})
            metrics = row.get("metrics", {})
            selected_count_value = first_number(
                controller.get("selected_count"),
                len(controller.get("context_item_ids_used", []) or []),
                len(controller.get("context_doc_ids_used", []) or []),
            )
            selected_doc_count_value = first_number(
                controller.get("selected_doc_count"),
                len(controller.get("context_doc_ids_used", []) or []),
            )
            theta_value = first_number(controller.get("theta_q"))
            context_tokens_value = first_number(controller.get("context_tokens_used"))
            budget_cap_value = first_number(runtime.get("budget_cap_tokens"), controller.get("budget_cap_tokens"))
            budget_source_value = runtime.get("budget_cap_source") or controller.get("budget_cap_source")
            ewma_prefill_value = first_number(runtime.get("ewma_prefill_ms_per_token"))
            coverage_goal_value = first_number(metrics.get("coverage_goal_met"))
            seed_size_value = first_number(controller.get("seed_item_count"))
            prefill_target_value = first_number(runtime.get("prefill_target_ms"))
            warmup_value = int(runtime.get("warmup_queries", 0) or 0)
        else:
            controller = row.get("context_budgeting", {})
            runtime = {}
            metrics = row.get("answer_metrics_per_query", {})
            selected_count_value = first_number(
                len(controller.get("context_chunk_ids_used", []) or []),
                controller.get("k_eff"),
            )
            selected_doc_count_value = first_number(len(controller.get("context_doc_ids_used", []) or []))
            theta_value = first_number(controller.get("query_local_theta"))
            context_tokens_value = first_number(controller.get("context_tokens_used"))
            budget_cap_value = first_number(controller.get("budget_cap_tokens"), controller.get("context_budget_tokens"))
            budget_source_value = controller.get("budget_cap_source")
            ewma_prefill_value = first_number(controller.get("ewma_prefill_ms_per_token"))
            coverage_goal_value = None
            seed_size_value = None
            prefill_target_value = first_number(controller.get("prefill_target_ms"))
            warmup_value = int(controller.get("warmup_queries", 0) or 0)

        warmup_queries = max(warmup_queries, warmup_value)
        if selected_count_value is not None:
            selected_count.append(float(selected_count_value))
        if selected_doc_count_value is not None:
            selected_doc_count.append(float(selected_doc_count_value))
        if seed_size_value is not None:
            seed_size.append(float(seed_size_value))
        if theta_value is not None:
            theta_values.append(float(theta_value))
        if coverage_goal_value is not None:
            coverage_goal_met.append(float(coverage_goal_value))
        if context_tokens_value is not None:
            context_tokens.append(float(context_tokens_value))
        if budget_cap_value is not None:
            budget_caps.append(float(budget_cap_value))
        if budget_source_value is not None:
            budget_sources.append(str(budget_source_value))
            if transition_query_index is None and str(budget_source_value) == "ewma_prefill":
                transition_query_index = idx
            if str(budget_source_value).startswith("bootstrap"):
                last_bootstrap_query_index = idx
        if ewma_prefill_value is not None:
            ewma_prefill.append(float(ewma_prefill_value))
        prefill_value = first_number(nested_get(row, "latency_ms", "t_prefill_ms"))
        if prefill_value is not None:
            t_prefill.append(float(prefill_value))
        if prefill_target_value is not None:
            prefill_targets.append(float(prefill_target_value))

        if idx >= warmup_value and prefill_value is not None and prefill_target_value is not None:
            post_warmup_count += 1
            abs_error = abs(float(prefill_value) - float(prefill_target_value))
            post_warmup_abs_errors.append(abs_error)
            if float(prefill_value) > (1.25 * float(prefill_target_value)):
                post_warmup_overshoots += 1

    summary_metrics = dict(summary_payload.get("metrics_mean", {}))
    if not summary_metrics:
        summary_metrics = aggregate_metrics(rows)
    summary_latency_ms = dict(summary_payload.get("latency_summary_ms", {}))

    post_warmup_caps = [
        float(first_number(
            nested_get(row, "runtime", "budget_cap_tokens"),
            nested_get(row, "context_controller", "budget_cap_tokens"),
            nested_get(row, "context_budgeting", "budget_cap_tokens"),
            nested_get(row, "context_budgeting", "context_budget_tokens"),
        ))
        for idx, row in enumerate(rows)
        if idx >= warmup_queries
        and first_number(
            nested_get(row, "runtime", "budget_cap_tokens"),
            nested_get(row, "context_controller", "budget_cap_tokens"),
            nested_get(row, "context_budgeting", "budget_cap_tokens"),
            nested_get(row, "context_budgeting", "context_budget_tokens"),
        ) is not None
    ]
    post_warmup_ewma = [
        float(first_number(
            nested_get(row, "runtime", "ewma_prefill_ms_per_token"),
            nested_get(row, "context_budgeting", "ewma_prefill_ms_per_token"),
        ))
        for idx, row in enumerate(rows)
        if idx >= warmup_queries
        and first_number(
            nested_get(row, "runtime", "ewma_prefill_ms_per_token"),
            nested_get(row, "context_budgeting", "ewma_prefill_ms_per_token"),
        ) is not None
    ]
    post_warmup_prefill = [
        float(first_number(nested_get(row, "latency_ms", "t_prefill_ms")))
        for idx, row in enumerate(rows)
        if idx >= warmup_queries and first_number(nested_get(row, "latency_ms", "t_prefill_ms")) is not None
    ]

    ewma_rel_delta_mean = None
    if len(post_warmup_ewma) >= 2:
        rel_deltas = []
        for prev, curr in zip(post_warmup_ewma, post_warmup_ewma[1:]):
            rel_deltas.append(abs(curr - prev) / max(1e-6, abs(prev)))
        ewma_rel_delta_mean = float(mean(rel_deltas))

    out = {
        "summary_metrics": summary_metrics,
        "summary_latency_ms": summary_latency_ms,
        "controller": {
            "selected_count": summarize(selected_count),
            "selected_doc_count": summarize(selected_doc_count),
            "seed_size": summarize(seed_size),
            "theta_q": summarize(theta_values),
            "theta_q_lt_0_05_count": int(sum(1 for value in theta_values if value < 0.05)),
            "theta_q_lt_0_05_ratio": float(sum(1 for value in theta_values if value < 0.05) / len(theta_values))
            if theta_values
            else 0.0,
            "coverage_goal_met_mean": float(mean(coverage_goal_met)) if coverage_goal_met else None,
            "context_tokens_used": summarize(context_tokens),
        },
        "hardware_adaptive": {
            "budget_cap_tokens": summarize(budget_caps),
            "budget_cap_source_counts": dict(Counter(budget_sources)),
            "warmup_queries": int(warmup_queries),
            "transition_query_index": transition_query_index,
            "last_bootstrap_query_index": last_bootstrap_query_index,
            "post_warmup_budget_cap_tokens": summarize(post_warmup_caps),
            "ewma_prefill_ms_per_token": summarize(ewma_prefill),
            "post_warmup_ewma_prefill_ms_per_token": summarize(post_warmup_ewma),
            "post_warmup_ewma_relative_delta_mean": ewma_rel_delta_mean,
        },
        "prefill_tracking": {
            "prefill_target_ms": summarize(prefill_targets),
            "t_prefill_ms": summarize(t_prefill),
            "post_warmup_t_prefill_ms": summarize(post_warmup_prefill),
            "abs_prefill_target_error_ms_post_warmup": summarize(post_warmup_abs_errors),
            "overshoot_rate_post_warmup": float(post_warmup_overshoots / max(1, post_warmup_count)),
            "post_warmup_query_count": int(post_warmup_count),
        },
    }

    payload = json.dumps(out, indent=2, ensure_ascii=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
