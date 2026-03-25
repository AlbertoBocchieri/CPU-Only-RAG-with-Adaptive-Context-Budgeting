#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from agnostic_cpu_rag.weight_search import LEGACY_INHERITED_WEIGHTS, PILOT75_ACCEPTABLE_F1_DROP

MULTI_HOP_CONFIRM_CI_FLOOR = -0.03
OPEN_QA_CONFIRM_F1_FLOOR = -0.02
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 42


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["qid"])] = row
    return rows


def nested_get(node: dict[str, Any], path: str) -> Any:
    cur: Any = node
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def metric_mean(summary: dict[str, Any], path: str) -> float:
    value = nested_get(summary, path)
    if value is None:
        return 0.0
    return float(value)


def summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "f1": metric_mean(summary, "metrics_mean.f1"),
        "em": metric_mean(summary, "metrics_mean.em"),
        "pair_in_context": metric_mean(summary, "metrics_mean.pair_in_context"),
        "relevant_doc_recall": metric_mean(summary, "metrics_mean.relevant_doc_recall"),
        "coverage_goal_met": metric_mean(summary, "metrics_mean.coverage_goal_met"),
        "selected_count_mean": metric_mean(summary, "context_controller_summary.selected_count.mean"),
        "context_mean": metric_mean(summary, "latency_summary_ms.context_tokens_used.mean"),
        "ttft_p50": metric_mean(summary, "latency_summary_ms.ttft_ms.p50"),
        "t_total_p50": metric_mean(summary, "latency_summary_ms.t_total_ms.p50"),
    }


def paired_mean_delta(base_rows: dict[str, dict[str, Any]], new_rows: dict[str, dict[str, Any]], metric_path: str) -> float:
    deltas: list[float] = []
    for qid in base_rows:
        if qid not in new_rows:
            continue
        base_value = nested_get(base_rows[qid], metric_path)
        new_value = nested_get(new_rows[qid], metric_path)
        if base_value is None or new_value is None:
            continue
        deltas.append(float(new_value) - float(base_value))
    if not deltas:
        return 0.0
    return float(sum(deltas) / len(deltas))


def paired_bootstrap_ci(
    base_rows: dict[str, dict[str, Any]],
    new_rows: dict[str, dict[str, Any]],
    *,
    metric_path: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    deltas: list[float] = []
    for qid in base_rows:
        if qid not in new_rows:
            continue
        base_value = nested_get(base_rows[qid], metric_path)
        new_value = nested_get(new_rows[qid], metric_path)
        if base_value is None or new_value is None:
            continue
        deltas.append(float(new_value) - float(base_value))
    if not deltas:
        return {"num_qids": 0, "mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(int(resamples)):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    low = boots[int(0.025 * (len(boots) - 1))]
    high = boots[int(0.975 * (len(boots) - 1))]
    return {
        "num_qids": len(deltas),
        "mean_delta": float(sum(deltas) / len(deltas)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def run_paths(validation_dir: Path, run_id: str, dataset: str) -> tuple[Path, Path]:
    run_dir = validation_dir / run_id / dataset
    return run_dir / "summary.json", run_dir / "per_query.jsonl"


def load_run(validation_dir: Path, run_id: str, dataset: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary_path, per_query_path = run_paths(validation_dir, run_id, dataset)
    return load_json(summary_path), load_jsonl(per_query_path)


def load_shortlist(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(load_json(path))


def candidate_run_available(validation_dir: Path, run_id: str, dataset: str) -> bool:
    summary_path, per_query_path = run_paths(validation_dir, run_id, dataset)
    return summary_path.exists() and per_query_path.exists()


def update_selection_report(path: Path, *, winner_reason: str | None, reject_reason: str | None, selection_stage: str, pilot75_metrics: dict[str, Any] | None = None, representative300_metrics: dict[str, Any] | None = None) -> None:
    report = load_json(path)
    report["winner_reason"] = winner_reason
    report["reject_reason"] = reject_reason
    report["selection_stage"] = selection_stage
    if pilot75_metrics is not None:
        report["pilot75_metrics"] = pilot75_metrics
    if representative300_metrics is not None:
        report["representative300_metrics"] = representative300_metrics
    atomic_write_json(path, report)


def write_combined_winner_yaml(path: Path, *, multi_hop_weights: dict[str, float], open_qa_weights: dict[str, float]) -> None:
    payload = {
        "task_profiles": {
            "multi_hop_qa": {"context_controller": {"utility_weights": dict(multi_hop_weights)}},
            "open_qa": {"context_controller": {"utility_weights": dict(open_qa_weights)}},
        }
    }
    atomic_write_yaml(path, payload)


def select_multi_hop(derived_dir: Path, validation_dir: Path, winners_dir: Path) -> dict[str, Any]:
    shortlist = load_shortlist(derived_dir / "multi_hop_qa_shortlist.json")
    inherited_summary_hotpot, inherited_rows_hotpot = load_run(validation_dir, "hotpot_pilot75_inherited", "hotpot_qa")
    inherited_summary_twowiki, inherited_rows_twowiki = load_run(validation_dir, "twowiki_pilot75_inherited", "two_wiki_multihop")
    inherited_metrics = {
        "hotpot_qa": summary_metrics(inherited_summary_hotpot),
        "two_wiki_multihop": summary_metrics(inherited_summary_twowiki),
    }

    candidate_rows: list[dict[str, Any]] = []
    for row in shortlist:
        cname = Path(row["override_path"]).stem
        if not candidate_run_available(validation_dir, f"hotpot_pilot75_{cname}", "hotpot_qa"):
            continue
        if not candidate_run_available(validation_dir, f"twowiki_pilot75_{cname}", "two_wiki_multihop"):
            continue
        hotpot_summary, hotpot_rows = load_run(validation_dir, f"hotpot_pilot75_{cname}", "hotpot_qa")
        twowiki_summary, twowiki_rows = load_run(validation_dir, f"twowiki_pilot75_{cname}", "two_wiki_multihop")
        delta_hotpot_f1 = paired_mean_delta(inherited_rows_hotpot, hotpot_rows, "metrics.f1")
        delta_twowiki_f1 = paired_mean_delta(inherited_rows_twowiki, twowiki_rows, "metrics.f1")
        candidate_rows.append(
            {
                "candidate_name": cname,
                "weights": dict(row["weights"]),
                "override_path": str(row["override_path"]),
                "l1_distance_to_anchor": float(row.get("l1_distance_to_anchor", 0.0)),
                "datasets": {
                    "hotpot_qa": {
                        "metrics": summary_metrics(hotpot_summary),
                        "delta_f1_vs_inherited": float(delta_hotpot_f1),
                    },
                    "two_wiki_multihop": {
                        "metrics": summary_metrics(twowiki_summary),
                        "delta_f1_vs_inherited": float(delta_twowiki_f1),
                    },
                },
                "worst_case_delta_f1": float(min(delta_hotpot_f1, delta_twowiki_f1)),
                "mean_delta_f1": float((delta_hotpot_f1 + delta_twowiki_f1) / 2.0),
                "mean_ttft_p50": float((summary_metrics(hotpot_summary)["ttft_p50"] + summary_metrics(twowiki_summary)["ttft_p50"]) / 2.0),
                "mean_context_mean": float((summary_metrics(hotpot_summary)["context_mean"] + summary_metrics(twowiki_summary)["context_mean"]) / 2.0),
            }
        )

    accepted = [row for row in candidate_rows if row["worst_case_delta_f1"] >= PILOT75_ACCEPTABLE_F1_DROP]
    accepted_sorted = sorted(
        accepted,
        key=lambda row: (
            -row["worst_case_delta_f1"],
            -row["mean_delta_f1"],
            row["mean_ttft_p50"],
            row["mean_context_mean"],
            row["l1_distance_to_anchor"],
            row["candidate_name"],
        ),
    )

    if accepted_sorted:
        winner = accepted_sorted[0]
        winner_type = "candidate"
        winner_reason = "pilot75_candidate_selected"
        reject_reason = None
        winner_override_path = str(winner["override_path"])
        winner_weights = dict(winner["weights"])
        winner_name = str(winner["candidate_name"])
    else:
        winner = None
        winner_type = "inherited"
        winner_reason = "inherited_retained"
        reject_reason = "no_candidate_passed_pilot75_threshold"
        winner_override_path = None
        winner_weights = dict(LEGACY_INHERITED_WEIGHTS)
        winner_name = "inherited"

    pilot75_metrics = {
        "inherited": inherited_metrics,
        "candidates": candidate_rows,
        "accepted_candidates": [row["candidate_name"] for row in accepted_sorted],
    }
    report = {
        "task_family": "multi_hop_qa",
        "winner_type": winner_type,
        "winner_name": winner_name,
        "winner_override_path": winner_override_path,
        "winner_weights": winner_weights,
        "winner_reason": winner_reason,
        "reject_reason": reject_reason,
        "representative300_status": "pending" if winner_type == "candidate" else "skipped_inherited_retained",
        "pilot75_metrics": pilot75_metrics,
        "representative300_metrics": {},
    }
    atomic_write_json(winners_dir / "multi_hop_winner_report.json", report)
    update_selection_report(
        derived_dir / "multi_hop_selection_report.json",
        winner_reason=winner_reason,
        reject_reason=reject_reason,
        selection_stage="pilot75_complete",
        pilot75_metrics=pilot75_metrics,
    )
    return report


def select_open_qa(derived_dir: Path, validation_dir: Path, winners_dir: Path) -> dict[str, Any]:
    shortlist = load_shortlist(derived_dir / "open_qa_shortlist.json")
    inherited_summary, inherited_rows = load_run(validation_dir, "squad_pilot75_inherited", "squad_open")
    no_controller_summary, _ = load_run(validation_dir, "squad_pilot75_no_controller", "squad_open")
    inherited_metrics = summary_metrics(inherited_summary)
    no_controller_metrics = summary_metrics(no_controller_summary)

    candidate_rows: list[dict[str, Any]] = []
    for row in shortlist:
        cname = Path(row["override_path"]).stem
        if not candidate_run_available(validation_dir, f"squad_pilot75_{cname}", "squad_open"):
            continue
        candidate_summary, candidate_rows_map = load_run(validation_dir, f"squad_pilot75_{cname}", "squad_open")
        candidate_metrics = summary_metrics(candidate_summary)
        delta_f1 = paired_mean_delta(inherited_rows, candidate_rows_map, "metrics.f1")
        passes = (
            delta_f1 >= PILOT75_ACCEPTABLE_F1_DROP
            and candidate_metrics["context_mean"] <= (0.70 * max(no_controller_metrics["context_mean"], 1e-9))
            and candidate_metrics["ttft_p50"] <= (0.70 * max(no_controller_metrics["ttft_p50"], 1e-9))
        )
        candidate_rows.append(
            {
                "candidate_name": cname,
                "weights": dict(row["weights"]),
                "override_path": str(row["override_path"]),
                "l1_distance_to_anchor": float(row.get("l1_distance_to_anchor", 0.0)),
                "metrics": candidate_metrics,
                "delta_f1_vs_inherited": float(delta_f1),
                "context_ratio_vs_no_controller": float(candidate_metrics["context_mean"] / max(no_controller_metrics["context_mean"], 1e-9)),
                "ttft_ratio_vs_no_controller": float(candidate_metrics["ttft_p50"] / max(no_controller_metrics["ttft_p50"], 1e-9)),
                "accepted": bool(passes),
            }
        )

    accepted = [row for row in candidate_rows if row["accepted"]]
    accepted_sorted = sorted(
        accepted,
        key=lambda row: (
            -row["delta_f1_vs_inherited"],
            row["metrics"]["ttft_p50"],
            row["metrics"]["context_mean"],
            row["l1_distance_to_anchor"],
            row["candidate_name"],
        ),
    )
    if accepted_sorted:
        winner = accepted_sorted[0]
        winner_type = "candidate"
        winner_reason = "pilot75_candidate_selected"
        reject_reason = None
        winner_override_path = str(winner["override_path"])
        winner_weights = dict(winner["weights"])
        winner_name = str(winner["candidate_name"])
    else:
        winner = None
        winner_type = "inherited"
        winner_reason = "inherited_retained"
        reject_reason = "no_candidate_passed_pilot75_threshold"
        winner_override_path = None
        winner_weights = dict(LEGACY_INHERITED_WEIGHTS)
        winner_name = "inherited"

    pilot75_metrics = {
        "inherited": inherited_metrics,
        "no_controller": no_controller_metrics,
        "candidates": candidate_rows,
        "accepted_candidates": [row["candidate_name"] for row in accepted_sorted],
    }
    report = {
        "task_family": "open_qa",
        "winner_type": winner_type,
        "winner_name": winner_name,
        "winner_override_path": winner_override_path,
        "winner_weights": winner_weights,
        "winner_reason": winner_reason,
        "reject_reason": reject_reason,
        "representative300_status": "pending" if winner_type == "candidate" else "skipped_inherited_retained",
        "pilot75_metrics": pilot75_metrics,
        "representative300_metrics": {},
    }
    atomic_write_json(winners_dir / "open_qa_winner_report.json", report)
    update_selection_report(
        derived_dir / "open_qa_selection_report.json",
        winner_reason=winner_reason,
        reject_reason=reject_reason,
        selection_stage="pilot75_complete",
        pilot75_metrics=pilot75_metrics,
    )
    return report


def update_multi_hop_confirm(derived_dir: Path, validation_dir: Path, winners_dir: Path) -> dict[str, Any]:
    report_path = winners_dir / "multi_hop_winner_report.json"
    report = load_json(report_path)
    if report.get("winner_type") != "candidate":
        report["representative300_status"] = "skipped_inherited_retained"
        atomic_write_json(report_path, report)
        update_selection_report(
            derived_dir / "multi_hop_selection_report.json",
            winner_reason=report.get("winner_reason"),
            reject_reason=report.get("reject_reason"),
            selection_stage="representative300_skipped_inherited_retained",
            representative300_metrics={},
        )
        return report

    winner_summary_hotpot, winner_rows_hotpot = load_run(validation_dir, "hotpot_rep300_winner", "hotpot_qa")
    inherited_summary_hotpot, inherited_rows_hotpot = load_run(validation_dir, "hotpot_rep300_inherited", "hotpot_qa")
    winner_summary_twowiki, winner_rows_twowiki = load_run(validation_dir, "twowiki_rep300_winner", "two_wiki_multihop")
    inherited_summary_twowiki, inherited_rows_twowiki = load_run(validation_dir, "twowiki_rep300_inherited", "two_wiki_multihop")

    hotpot_ci = paired_bootstrap_ci(inherited_rows_hotpot, winner_rows_hotpot, metric_path="metrics.f1")
    twowiki_ci = paired_bootstrap_ci(inherited_rows_twowiki, winner_rows_twowiki, metric_path="metrics.f1")
    hotpot_metrics = summary_metrics(winner_summary_hotpot)
    hotpot_inherited_metrics = summary_metrics(inherited_summary_hotpot)
    twowiki_metrics = summary_metrics(winner_summary_twowiki)
    twowiki_inherited_metrics = summary_metrics(inherited_summary_twowiki)

    hotpot_pass = (
        hotpot_ci["ci95_low"] > MULTI_HOP_CONFIRM_CI_FLOOR
        and hotpot_metrics["ttft_p50"] <= (1.10 * max(hotpot_inherited_metrics["ttft_p50"], 1e-9))
    )
    twowiki_pass = (
        twowiki_ci["ci95_low"] > MULTI_HOP_CONFIRM_CI_FLOOR
        and twowiki_metrics["pair_in_context"] >= (twowiki_inherited_metrics["pair_in_context"] - 0.01)
        and twowiki_metrics["relevant_doc_recall"] >= (twowiki_inherited_metrics["relevant_doc_recall"] - 0.01)
    )
    report["representative300_metrics"] = {
        "hotpot_qa": {
            "winner": hotpot_metrics,
            "inherited": hotpot_inherited_metrics,
            "delta_f1": paired_mean_delta(inherited_rows_hotpot, winner_rows_hotpot, "metrics.f1"),
            "bootstrap_f1": hotpot_ci,
            "passed": bool(hotpot_pass),
        },
        "two_wiki_multihop": {
            "winner": twowiki_metrics,
            "inherited": twowiki_inherited_metrics,
            "delta_f1": paired_mean_delta(inherited_rows_twowiki, winner_rows_twowiki, "metrics.f1"),
            "bootstrap_f1": twowiki_ci,
            "passed": bool(twowiki_pass),
        },
    }
    report["representative300_status"] = "passed" if (hotpot_pass and twowiki_pass) else "failed"
    if report["representative300_status"] == "failed":
        report["reject_reason"] = "representative300_gate_failed"
    atomic_write_json(report_path, report)
    update_selection_report(
        derived_dir / "multi_hop_selection_report.json",
        winner_reason=report.get("winner_reason"),
        reject_reason=report.get("reject_reason"),
        selection_stage="representative300_complete",
        representative300_metrics=report["representative300_metrics"],
    )
    return report


def update_open_qa_confirm(derived_dir: Path, validation_dir: Path, winners_dir: Path) -> dict[str, Any]:
    report_path = winners_dir / "open_qa_winner_report.json"
    report = load_json(report_path)
    if report.get("winner_type") != "candidate":
        report["representative300_status"] = "skipped_inherited_retained"
        atomic_write_json(report_path, report)
        update_selection_report(
            derived_dir / "open_qa_selection_report.json",
            winner_reason=report.get("winner_reason"),
            reject_reason=report.get("reject_reason"),
            selection_stage="representative300_skipped_inherited_retained",
            representative300_metrics={},
        )
        return report

    winner_summary, winner_rows = load_run(validation_dir, "squad_rep300_winner", "squad_open")
    inherited_summary, inherited_rows = load_run(validation_dir, "squad_rep300_inherited", "squad_open")
    no_controller_summary, _ = load_run(validation_dir, "squad_rep300_no_controller", "squad_open")
    winner_metrics = summary_metrics(winner_summary)
    inherited_metrics = summary_metrics(inherited_summary)
    no_controller_metrics = summary_metrics(no_controller_summary)
    delta_f1 = paired_mean_delta(inherited_rows, winner_rows, "metrics.f1")
    passed = (
        delta_f1 >= OPEN_QA_CONFIRM_F1_FLOOR
        and winner_metrics["context_mean"] <= (0.70 * max(no_controller_metrics["context_mean"], 1e-9))
        and winner_metrics["ttft_p50"] <= (0.70 * max(no_controller_metrics["ttft_p50"], 1e-9))
    )
    report["representative300_metrics"] = {
        "squad_open": {
            "winner": winner_metrics,
            "inherited": inherited_metrics,
            "no_controller": no_controller_metrics,
            "delta_f1": float(delta_f1),
            "passed": bool(passed),
        }
    }
    report["representative300_status"] = "passed" if passed else "failed"
    if report["representative300_status"] == "failed":
        report["reject_reason"] = "representative300_gate_failed"
    atomic_write_json(report_path, report)
    update_selection_report(
        derived_dir / "open_qa_selection_report.json",
        winner_reason=report.get("winner_reason"),
        reject_reason=report.get("reject_reason"),
        selection_stage="representative300_complete",
        representative300_metrics=report["representative300_metrics"],
    )
    return report


def run_pilot75(args: argparse.Namespace) -> None:
    derived_dir = Path(args.derived_dir)
    validation_dir = Path(args.validation_dir)
    winners_dir = Path(args.output_dir)
    winners_dir.mkdir(parents=True, exist_ok=True)

    multi_hop_report = select_multi_hop(derived_dir, validation_dir, winners_dir)
    open_qa_report = select_open_qa(derived_dir, validation_dir, winners_dir)
    write_combined_winner_yaml(
        winners_dir / "task_family_weights_winner.yaml",
        multi_hop_weights=dict(multi_hop_report["winner_weights"]),
        open_qa_weights=dict(open_qa_report["winner_weights"]),
    )


def run_confirm300(args: argparse.Namespace) -> None:
    derived_dir = Path(args.derived_dir)
    validation_dir = Path(args.validation_dir)
    winners_dir = Path(args.output_dir)
    multi_hop_report = update_multi_hop_confirm(derived_dir, validation_dir, winners_dir)
    open_qa_report = update_open_qa_confirm(derived_dir, validation_dir, winners_dir)
    write_combined_winner_yaml(
        winners_dir / "task_family_weights_winner.yaml",
        multi_hop_weights=dict(multi_hop_report["winner_weights"]),
        open_qa_weights=dict(open_qa_report["winner_weights"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select or confirm task-family utility-weight winners.")
    sub = parser.add_subparsers(dest="mode", required=True)

    pilot = sub.add_parser("pilot75")
    pilot.add_argument("--derived-dir", required=True)
    pilot.add_argument("--validation-dir", required=True)
    pilot.add_argument("--output-dir", required=True)

    confirm = sub.add_parser("confirm300")
    confirm.add_argument("--derived-dir", required=True)
    confirm.add_argument("--validation-dir", required=True)
    confirm.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.mode == "pilot75":
        run_pilot75(args)
    else:
        run_confirm300(args)


if __name__ == "__main__":
    main()
