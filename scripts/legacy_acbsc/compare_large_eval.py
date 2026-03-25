#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["qid"])] = row
    return rows


def nested_get(record: dict[str, Any], path: str) -> Any:
    node: Any = record
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def paired_delta(base_rows: dict[str, dict[str, Any]], new_rows: dict[str, dict[str, Any]], metric_path: str) -> dict[str, Any]:
    qids = [qid for qid in base_rows if qid in new_rows]
    deltas: list[float] = []
    better = worse = ties = 0
    for qid in qids:
        base_value = nested_get(base_rows[qid], metric_path)
        new_value = nested_get(new_rows[qid], metric_path)
        if base_value is None or new_value is None:
            continue
        delta = float(new_value) - float(base_value)
        deltas.append(delta)
        better += int(delta > 0)
        worse += int(delta < 0)
        ties += int(delta == 0)
    rng = random.Random(42)
    boots: list[float] = []
    if deltas:
        for _ in range(10000):
            sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
            boots.append(sum(sample) / len(sample))
        boots.sort()
    mean_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    return {
        "num_qids": int(len(deltas)),
        "mean_delta": float(mean_delta),
        "ci95_low": float(quantile(boots, 0.025)) if boots else 0.0,
        "ci95_high": float(quantile(boots, 0.975)) if boots else 0.0,
        "better": int(better),
        "worse": int(worse),
        "tie": int(ties),
    }


def arm_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "EM": float(nested_get(summary, "generation.EM") or 0.0),
        "F1": float(nested_get(summary, "generation.F1") or 0.0),
        "pair_in_context_at_k_mean": nested_get(summary, "post_context.pair_in_context_at_k_mean"),
        "support_doc_recall@10": float(nested_get(summary, "supporting_docs.support_doc_recall.support_doc_recall@10") or 0.0),
        "context_tokens_used_mean": float(nested_get(summary, "context_budgeting.context_tokens_used.mean") or 0.0),
        "k_eff_mean": float(nested_get(summary, "context_budgeting.k_eff.mean") or 0.0),
        "ttft_p50_ms": float(nested_get(summary, "latency_ms.ttft_ms.p50") or 0.0),
        "t_total_p50_ms": float(nested_get(summary, "latency_ms.t_total_ms.p50") or 0.0),
    }


def resolve_summary(run_root: Path, dataset: str) -> tuple[Path, Path, dict[str, Any], dict[str, dict[str, Any]]]:
    cfg_dirs = sorted(run_root.glob("cfg_*"))
    if len(cfg_dirs) != 1:
        raise ValueError(f"Expected exactly one cfg_* directory under {run_root}, found {len(cfg_dirs)}")
    cfg_dir = cfg_dirs[0]
    summary_path = cfg_dir / dataset / "summary.json"
    summary = load_json(summary_path)
    per_query_path = Path(str(nested_get(summary, "artifacts.per_query_jsonl") or (cfg_dir / dataset / "per_query.jsonl")))
    return summary_path, per_query_path, summary, load_rows(per_query_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare explicit large-pool baseline vs ACB-SC runs.")
    parser.add_argument("--baseline-run-root", required=True)
    parser.add_argument("--candidate-run-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    baseline_summary_path, baseline_per_query_path, baseline_summary, baseline_rows = resolve_summary(Path(args.baseline_run_root), args.dataset)
    candidate_summary_path, candidate_per_query_path, candidate_summary, candidate_rows = resolve_summary(Path(args.candidate_run_root), args.dataset)

    report = {
        "dataset": args.dataset,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "arms": {
            "baseline": {
                "summary_path": str(baseline_summary_path),
                "per_query_path": str(baseline_per_query_path),
                "metrics": arm_metrics(baseline_summary),
            },
            "candidate": {
                "summary_path": str(candidate_summary_path),
                "per_query_path": str(candidate_per_query_path),
                "metrics": arm_metrics(candidate_summary),
            },
        },
        "paired": {
            "delta_f1": paired_delta(baseline_rows, candidate_rows, "answer_metrics_per_query.f1"),
            "delta_em": paired_delta(baseline_rows, candidate_rows, "answer_metrics_per_query.em"),
        },
    }

    output_path = Path(args.output_dir) / args.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
