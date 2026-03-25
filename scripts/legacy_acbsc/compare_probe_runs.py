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
    with path.open(encoding="utf-8") as handle:
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



def discover_arms(run_root: Path, dataset: str) -> dict[str, dict[str, Any]]:
    arms: dict[str, dict[str, Any]] = {}
    for cfg_dir in sorted(run_root.glob("cfg_*")):
        effective_config = load_json(cfg_dir / "effective_config.json")
        arm_name = classify_arm(effective_config)
        summary_path = cfg_dir / dataset / "summary.json"
        summary = load_json(summary_path)
        per_query_path = Path(str(nested_get(summary, "artifacts.per_query_jsonl") or (cfg_dir / dataset / "per_query.jsonl")))
        arms[arm_name] = {
            "config_id": str(summary.get("config_id", cfg_dir.name.replace("cfg_", ""))),
            "summary_path": str(summary_path),
            "per_query_path": str(per_query_path),
            "summary": summary,
            "rows": load_rows(per_query_path),
            "metrics": arm_metrics(summary),
        }
    expected = {"no_acb", "legacy_incremental_sc", "new_agnostic_acb_sc"}
    missing = expected.difference(arms)
    if missing:
        raise ValueError(f"Missing probe arms under {run_root}: {sorted(missing)}")
    return arms



def evaluate_gates(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    legacy = arms["legacy_incremental_sc"]["metrics"]
    no_acb = arms["no_acb"]["metrics"]
    new = arms["new_agnostic_acb_sc"]["metrics"]
    paired_new_vs_legacy_f1 = paired_delta(
        arms["legacy_incremental_sc"]["rows"],
        arms["new_agnostic_acb_sc"]["rows"],
        "answer_metrics_per_query.f1",
    )
    paired_new_vs_legacy_em = paired_delta(
        arms["legacy_incremental_sc"]["rows"],
        arms["new_agnostic_acb_sc"]["rows"],
        "answer_metrics_per_query.em",
    )
    paired_new_vs_noacb_f1 = paired_delta(
        arms["no_acb"]["rows"],
        arms["new_agnostic_acb_sc"]["rows"],
        "answer_metrics_per_query.f1",
    )
    paired_new_vs_noacb_em = paired_delta(
        arms["no_acb"]["rows"],
        arms["new_agnostic_acb_sc"]["rows"],
        "answer_metrics_per_query.em",
    )
    primary_gate = {
        "delta_f1_floor": bool(paired_new_vs_legacy_f1["mean_delta"] >= -0.02),
        "pair_in_context_floor": bool(float(new["pair_in_context_at_k_mean"] or 0.0) >= float(legacy["pair_in_context_at_k_mean"] or 0.0) - 0.03),
        "support_doc_recall_floor": bool(float(new["support_doc_recall@10"]) >= float(legacy["support_doc_recall@10"]) - 0.03),
        "ttft_guard": bool(float(new["ttft_p50_ms"]) <= float(legacy["ttft_p50_ms"]) * 1.10),
    }
    secondary_material_improvement = any(
        [
            float(new["pair_in_context_at_k_mean"] or 0.0) > float(no_acb["pair_in_context_at_k_mean"] or 0.0),
            float(new["support_doc_recall@10"]) > float(no_acb["support_doc_recall@10"]),
            paired_new_vs_noacb_f1["mean_delta"] > 0.0,
        ]
    )
    secondary_gate = {
        "material_improvement": bool(secondary_material_improvement),
        "context_guard": bool(float(new["context_tokens_used_mean"]) <= float(no_acb["context_tokens_used_mean"]) * 1.15),
        "ttft_guard": bool(float(new["ttft_p50_ms"]) <= float(no_acb["ttft_p50_ms"]) * 1.15),
    }
    return {
        "new_vs_legacy_incremental_sc": {
            "delta_f1": paired_new_vs_legacy_f1,
            "delta_em": paired_new_vs_legacy_em,
            "passed": bool(all(primary_gate.values())),
            "checks": primary_gate,
        },
        "new_vs_no_acb": {
            "delta_f1": paired_new_vs_noacb_f1,
            "delta_em": paired_new_vs_noacb_em,
            "passed": bool(all(secondary_gate.values())),
            "checks": secondary_gate,
        },
    }



def compare_dataset(run_root: Path, dataset: str, output_path: Path) -> dict[str, Any]:
    arms = discover_arms(run_root, dataset)
    report = {
        "dataset": dataset,
        "run_root": str(run_root),
        "arms": {
            arm_name: {
                "config_id": payload["config_id"],
                "summary_path": payload["summary_path"],
                "per_query_path": payload["per_query_path"],
                "metrics": payload["metrics"],
            }
            for arm_name, payload in arms.items()
        },
        "paired": evaluate_gates(arms),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return report



def build_cross_dataset_summary(hotpot: dict[str, Any], twowiki: dict[str, Any], output_path: Path) -> dict[str, Any]:
    hotpot_primary = bool(nested_get(hotpot, "paired.new_vs_legacy_incremental_sc.passed"))
    twowiki_primary = bool(nested_get(twowiki, "paired.new_vs_legacy_incremental_sc.passed"))
    summary = {
        "datasets": {
            "hotpot_qa": hotpot,
            "two_wiki_multihop": twowiki,
        },
        "cross_dataset_portability_claim_allowed": bool(hotpot_primary and twowiki_primary),
        "hotpot_primary_passed": bool(hotpot_primary),
        "twowiki_primary_passed": bool(twowiki_primary),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary



def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 3-arm legacy ACB-SC probe runs.")
    parser.add_argument("--hotpot-run-root", required=True)
    parser.add_argument("--twowiki-run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-prefix", default="legacy_acbsc_probe")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    prefix = str(args.artifact_prefix).strip() or "legacy_acbsc_probe"
    hotpot_report = compare_dataset(
        run_root=Path(args.hotpot_run_root),
        dataset="hotpot_qa",
        output_path=output_dir / f"hotpot_{prefix}_compare.json",
    )
    twowiki_report = compare_dataset(
        run_root=Path(args.twowiki_run_root),
        dataset="two_wiki_multihop",
        output_path=output_dir / f"twowiki_{prefix}_compare.json",
    )
    build_cross_dataset_summary(
        hotpot=hotpot_report,
        twowiki=twowiki_report,
        output_path=output_dir / f"{prefix}_cross_dataset_summary.json",
    )


if __name__ == "__main__":
    main()
