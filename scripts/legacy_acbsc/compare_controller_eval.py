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


def classify_arm(cfg: dict[str, Any]) -> str:
    cb = cfg.get("context_budgeting", {})
    if not bool(cb.get("enabled", False)):
        return "no_acb"
    strategy = str(cb.get("strategy", "v1")).strip().lower()
    if strategy == "incremental_sc":
        return "legacy_incremental_sc"
    if strategy == "agnostic_acb_sc":
        return "new_agnostic_acb_sc"
    raise ValueError(f"Unsupported probe config strategy: {strategy}")


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


def load_explicit_arm(summary_path: Path, per_query_path: Path, label: str) -> dict[str, Any]:
    summary = load_json(summary_path)
    return {
        "label": label,
        "config_id": str(summary.get("config_id", "")),
        "summary_path": str(summary_path),
        "per_query_path": str(per_query_path),
        "summary": summary,
        "rows": load_rows(per_query_path),
        "metrics": arm_metrics(summary),
    }


def discover_run_arms(run_root: Path, dataset: str, expected: set[str]) -> dict[str, dict[str, Any]]:
    arms: dict[str, dict[str, Any]] = {}
    for cfg_dir in sorted(run_root.glob("cfg_*")):
        effective_config = load_json(cfg_dir / "effective_config.json")
        arm_name = classify_arm(effective_config)
        if arm_name not in expected:
            continue
        summary_path = cfg_dir / dataset / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        per_query_path = Path(str(nested_get(summary, "artifacts.per_query_jsonl") or (cfg_dir / dataset / "per_query.jsonl")))
        arms[arm_name] = {
            "label": arm_name,
            "config_id": str(summary.get("config_id", cfg_dir.name.replace("cfg_", ""))),
            "summary_path": str(summary_path),
            "per_query_path": str(per_query_path),
            "summary": summary,
            "rows": load_rows(per_query_path),
            "metrics": arm_metrics(summary),
        }
    missing = expected.difference(arms)
    if missing:
        raise ValueError(f"Missing expected arms under {run_root}: {sorted(missing)}")
    return arms


def evaluate_primary_gate(base_arm: dict[str, Any], new_arm: dict[str, Any]) -> dict[str, Any]:
    delta_f1 = paired_delta(base_arm["rows"], new_arm["rows"], "answer_metrics_per_query.f1")
    delta_em = paired_delta(base_arm["rows"], new_arm["rows"], "answer_metrics_per_query.em")
    base_metrics = base_arm["metrics"]
    new_metrics = new_arm["metrics"]
    checks = {
        "delta_f1_floor": bool(delta_f1["mean_delta"] >= -0.02),
        "pair_in_context_floor": bool(
            float(new_metrics["pair_in_context_at_k_mean"] or 0.0)
            >= float(base_metrics["pair_in_context_at_k_mean"] or 0.0) - 0.03
        ),
        "support_doc_recall_floor": bool(
            float(new_metrics["support_doc_recall@10"])
            >= float(base_metrics["support_doc_recall@10"]) - 0.03
        ),
        "ttft_guard": bool(
            float(new_metrics["ttft_p50_ms"]) <= float(base_metrics["ttft_p50_ms"]) * 1.10
        ),
    }
    return {
        "delta_f1": delta_f1,
        "delta_em": delta_em,
        "passed": bool(all(checks.values())),
        "checks": checks,
    }


def write_report(dataset: str, baseline_arm: dict[str, Any], new_arm: dict[str, Any], output_path: Path) -> dict[str, Any]:
    report = {
        "dataset": dataset,
        "baseline_label": str(baseline_arm["label"]),
        "candidate_label": str(new_arm["label"]),
        "arms": {
            "baseline": {
                "config_id": baseline_arm["config_id"],
                "summary_path": baseline_arm["summary_path"],
                "per_query_path": baseline_arm["per_query_path"],
                "metrics": baseline_arm["metrics"],
            },
            "candidate": {
                "config_id": new_arm["config_id"],
                "summary_path": new_arm["summary_path"],
                "per_query_path": new_arm["per_query_path"],
                "metrics": new_arm["metrics"],
            },
        },
        "paired": evaluate_primary_gate(baseline_arm, new_arm),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return report


def write_cross_summary(hotpot_report: dict[str, Any], twowiki_report: dict[str, Any], output_path: Path) -> dict[str, Any]:
    hotpot_primary = bool(nested_get(hotpot_report, "paired.passed"))
    twowiki_primary = bool(nested_get(twowiki_report, "paired.passed"))
    summary = {
        "datasets": {
            "hotpot_qa": hotpot_report,
            "two_wiki_multihop": twowiki_report,
        },
        "cross_dataset_portability_claim_allowed": bool(hotpot_primary and twowiki_primary),
        "hotpot_primary_passed": bool(hotpot_primary),
        "twowiki_primary_passed": bool(twowiki_primary),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare controller-only legacy-stack ACB-SC evaluations.")
    parser.add_argument("--hotpot-baseline-summary", required=True)
    parser.add_argument("--hotpot-baseline-per-query", required=True)
    parser.add_argument("--hotpot-candidate-run-root", required=True)
    parser.add_argument("--twowiki-run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    hotpot_baseline = load_explicit_arm(
        summary_path=Path(args.hotpot_baseline_summary),
        per_query_path=Path(args.hotpot_baseline_per_query),
        label="legacy_incremental_sc_existing_q1",
    )
    hotpot_candidate = discover_run_arms(
        run_root=Path(args.hotpot_candidate_run_root),
        dataset="hotpot_qa",
        expected={"new_agnostic_acb_sc"},
    )["new_agnostic_acb_sc"]
    twowiki_arms = discover_run_arms(
        run_root=Path(args.twowiki_run_root),
        dataset="two_wiki_multihop",
        expected={"legacy_incremental_sc", "new_agnostic_acb_sc"},
    )

    hotpot_report = write_report(
        dataset="hotpot_qa",
        baseline_arm=hotpot_baseline,
        new_arm=hotpot_candidate,
        output_path=output_dir / "hotpot_legacy_acbsc_vs_q1_rep300_compare.json",
    )
    twowiki_report = write_report(
        dataset="two_wiki_multihop",
        baseline_arm=twowiki_arms["legacy_incremental_sc"],
        new_arm=twowiki_arms["new_agnostic_acb_sc"],
        output_path=output_dir / "twowiki_legacy_acbsc_vs_q1_holdout1000_compare.json",
    )
    write_cross_summary(
        hotpot_report=hotpot_report,
        twowiki_report=twowiki_report,
        output_path=output_dir / "legacy_acbsc_controller_eval_cross_dataset_summary.json",
    )


if __name__ == "__main__":
    main()
