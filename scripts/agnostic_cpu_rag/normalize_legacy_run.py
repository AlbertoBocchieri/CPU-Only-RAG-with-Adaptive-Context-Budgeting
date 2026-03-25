#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agnostic_cpu_rag.evaluation import summarize_query_records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    answer = dict(row.get("answer_metrics_per_query", {}) or {})
    support = dict(answer.get("support_doc_metrics", {}) or {})
    pair = dict(answer.get("pair_recall_metrics", {}) or {})
    ctx = dict(row.get("context_budgeting", {}) or {})
    post = dict(row.get("post_context", {}) or {})
    selected_doc_ids = list(post.get("context_doc_ids_used") or ctx.get("context_doc_ids_used") or [])
    selected_item_ids = list(post.get("context_chunk_ids_used") or ctx.get("context_chunk_ids_used") or [])

    metrics = {
        "em": answer.get("em"),
        "f1": answer.get("f1"),
        "pair_in_context": post.get("pair_in_context_at_k"),
        "support_doc_in_context_at_2": post.get("support_doc_in_context_at_2"),
        "retrieval_support_doc_recall_at_10": support.get("support_doc_recall@10"),
        "retrieval_pair_recall_at_10": pair.get("pair_recall@10"),
    }

    context_controller = {
        "enabled": bool(ctx.get("enabled", False)),
        "strategy": ctx.get("strategy"),
        "stop_mode": ctx.get("stop_mode"),
        "selected_count": len(selected_item_ids),
        "selected_doc_count": len(selected_doc_ids),
        "context_tokens_used": ctx.get("context_tokens_used"),
        "budget_cap_tokens": first_float(ctx.get("budget_cap_tokens"), ctx.get("context_budget_tokens")),
        "budget_cap_source": ctx.get("budget_cap_source"),
        "theta_q": ctx.get("query_local_theta"),
    }

    runtime = {
        "query_count": (int(row.get("query_index", -1)) + 1) if row.get("query_index") is not None else None,
        "prefill_target_ms": ctx.get("prefill_target_ms"),
        "warmup_queries": ctx.get("warmup_queries"),
        "ewma_prefill_ms_per_token": ctx.get("ewma_prefill_ms_per_token"),
        "budget_cap_tokens": first_float(ctx.get("budget_cap_tokens"), ctx.get("context_budget_tokens")),
        "budget_cap_source": ctx.get("budget_cap_source"),
    }

    return {
        "qid": row.get("qid"),
        "query": row.get("question"),
        "prediction": row.get("prediction"),
        "retrieved_doc_ids": row.get("retrieved_doc_ids") or row.get("retrieved_ids") or [],
        "selected_doc_ids": selected_doc_ids,
        "latency_ms": dict(row.get("latency_ms", {}) or {}),
        "context_controller": context_controller,
        "runtime": runtime,
        "metrics": {k: v for k, v in metrics.items() if v is not None},
        "legacy": {
            "dataset": row.get("dataset"),
            "tier": row.get("tier"),
            "run_id": row.get("run_id"),
            "config_id": row.get("config_id"),
            "failed": row.get("failed"),
            "abstained": row.get("abstained"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize legacy rag_cpu per_query output to the agnostic schema.")
    parser.add_argument("per_query_jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    in_path = Path(args.per_query_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_rows = [normalize_row(row) for row in load_jsonl(in_path)]
    summary = summarize_query_records(normalized_rows)
    if args.summary_json:
        legacy_summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
        summary["legacy_summary_path"] = str(args.summary_json)
        summary["legacy_summary_metrics"] = dict(legacy_summary.get("answer_metrics", {}))

    (out_dir / "per_query.normalized.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in normalized_rows),
        encoding="utf-8",
    )
    (out_dir / "summary.normalized.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "source_per_query": str(in_path),
        "source_summary": str(args.summary_json) if args.summary_json else "",
        "num_queries": len(normalized_rows),
        "artifacts": {
            "per_query_normalized": str(out_dir / "per_query.normalized.jsonl"),
            "summary_normalized": str(out_dir / "summary.normalized.json"),
        },
    }
    (out_dir / "normalization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
