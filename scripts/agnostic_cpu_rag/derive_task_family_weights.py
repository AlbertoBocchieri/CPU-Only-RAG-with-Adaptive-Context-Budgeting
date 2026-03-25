#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from agnostic_cpu_rag.weight_search import (
    ANCHOR_SEARCH_RADIUS,
    ANCHOR_SEARCH_STEP,
    LEGACY_INHERITED_WEIGHTS,
    MULTI_HOP_LOCAL_BOUNDS,
    OPEN_QA_LOCAL_BOUNDS,
    anchor_distance_rank_key,
    build_anchor_local_grid,
    canonicalize_weights,
    evaluate_controller_cache,
    l1_distance_to_anchor,
    load_jsonl,
    metric_mean,
    open_qa_rank_key,
    pooled_multi_hop_rank_key,
    passes_open_qa_relative_gate,
    true_lodo_relative_filter,
    weight_signature,
)


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


def load_controller_cfg(path: str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if isinstance(payload, dict) and isinstance(payload.get("context_controller"), dict):
        return dict(payload.get("context_controller", {}) or {})
    return dict(payload or {})


def summarize_candidate(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "pair_in_context": metric_mean(summary, "pair_in_context"),
        "relevant_doc_recall": metric_mean(summary, "relevant_doc_recall"),
        "coverage_goal_met": metric_mean(summary, "coverage_goal_met"),
        "context_mean": metric_mean(summary, "context_tokens_used"),
        "selected_count_mean": metric_mean(summary, "selected_count"),
    }


def delta_vs_baseline(summary: dict[str, Any], baseline_summary: dict[str, Any]) -> dict[str, float]:
    return {
        "pair_in_context": metric_mean(summary, "pair_in_context") - metric_mean(baseline_summary, "pair_in_context"),
        "relevant_doc_recall": metric_mean(summary, "relevant_doc_recall") - metric_mean(baseline_summary, "relevant_doc_recall"),
        "coverage_goal_met": metric_mean(summary, "coverage_goal_met") - metric_mean(baseline_summary, "coverage_goal_met"),
        "context_mean": metric_mean(summary, "context_tokens_used") - metric_mean(baseline_summary, "context_tokens_used"),
        "selected_count_mean": metric_mean(summary, "selected_count") - metric_mean(baseline_summary, "selected_count"),
    }


def write_candidate_overrides(
    *,
    out_dir: Path,
    task_family: str,
    ranked_signatures: list[str],
    score_rows: dict[str, dict[str, Any]],
    extra_rows: dict[str, dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    shortlist: list[dict[str, Any]] = []
    for idx, sig in enumerate(ranked_signatures[:top_n], start=1):
        weights = dict(score_rows[sig]["weights"])
        payload = {
            "task_profiles": {
                task_family: {
                    "context_controller": {
                        "utility_weights": weights,
                    }
                }
            }
        }
        override_path = out_dir / f"{task_family}_candidate_{idx}.yaml"
        atomic_write_yaml(override_path, payload)
        shortlist.append(
            {
                "rank": idx,
                "signature": sig,
                "weights": weights,
                "override_path": str(override_path),
                **extra_rows[sig],
            }
        )
    atomic_write_json(out_dir / f"{task_family}_shortlist.json", shortlist)
    return shortlist


def run_multi_hop(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    selection_path = out_dir / "multi_hop_selection_report.json"
    candidates_path = out_dir / "multi_hop_local_candidates.json"
    hotpot_rows = load_jsonl(args.hotpot_cache)
    twowiki_rows = load_jsonl(args.twowiki_cache)
    controller_cfg = load_controller_cfg(args.controller_cfg_yaml)

    grid = build_anchor_local_grid(
        step=ANCHOR_SEARCH_STEP,
        anchor=LEGACY_INHERITED_WEIGHTS,
        radius=ANCHOR_SEARCH_RADIUS,
        bounds=MULTI_HOP_LOCAL_BOUNDS,
        include=[LEGACY_INHERITED_WEIGHTS],
    )

    scores: dict[str, dict[str, Any]] = {}
    export_rows: list[dict[str, Any]] = []
    for weights in grid:
        sig = weight_signature(weights)
        hotpot_eval = evaluate_controller_cache(hotpot_rows, task_family="multi_hop_qa", controller_cfg=controller_cfg, weights=weights)
        twowiki_eval = evaluate_controller_cache(twowiki_rows, task_family="multi_hop_qa", controller_cfg=controller_cfg, weights=weights)
        scores[sig] = {
            "weights": canonicalize_weights(weights),
            "hotpot_qa": hotpot_eval["summary"],
            "two_wiki_multihop": twowiki_eval["summary"],
        }

    inherited_sig = weight_signature(LEGACY_INHERITED_WEIGHTS)
    inherited_hotpot = scores[inherited_sig]["hotpot_qa"]
    inherited_twowiki = scores[inherited_sig]["two_wiki_multihop"]
    filter_result = true_lodo_relative_filter(
        scores,
        hotpot_baseline=inherited_hotpot,
        twowiki_baseline=inherited_twowiki,
    )

    extra_rows: dict[str, dict[str, Any]] = {}
    for sig, row in scores.items():
        l1 = l1_distance_to_anchor(row["weights"], LEGACY_INHERITED_WEIGHTS)
        hotpot_metrics = summarize_candidate(row["hotpot_qa"])
        twowiki_metrics = summarize_candidate(row["two_wiki_multihop"])
        export_row = {
            "weight_signature": sig,
            "weights": dict(row["weights"]),
            "l1_distance_to_anchor": l1,
            "passes_filter": sig in set(filter_result["survivors"]),
            "hotpot_qa": {
                "metrics": hotpot_metrics,
                "delta_vs_inherited": delta_vs_baseline(row["hotpot_qa"], inherited_hotpot),
            },
            "two_wiki_multihop": {
                "metrics": twowiki_metrics,
                "delta_vs_inherited": delta_vs_baseline(row["two_wiki_multihop"], inherited_twowiki),
            },
        }
        export_rows.append(export_row)
        extra_rows[sig] = {
            "l1_distance_to_anchor": l1,
            "metrics": {
                "hotpot_qa": hotpot_metrics,
                "two_wiki_multihop": twowiki_metrics,
            },
            "delta_vs_inherited": {
                "hotpot_qa": delta_vs_baseline(row["hotpot_qa"], inherited_hotpot),
                "two_wiki_multihop": delta_vs_baseline(row["two_wiki_multihop"], inherited_twowiki),
            },
        }
    atomic_write_json(candidates_path, export_rows)

    ranked_survivors = sorted(
        [sig for sig in filter_result["survivors"] if sig != inherited_sig],
        key=lambda sig: (
            anchor_distance_rank_key(scores[sig]["weights"], anchor=LEGACY_INHERITED_WEIGHTS),
            pooled_multi_hop_rank_key(scores[sig]["hotpot_qa"], scores[sig]["two_wiki_multihop"]),
        ),
    )
    shortlist = write_candidate_overrides(
        out_dir=out_dir,
        task_family="multi_hop_qa",
        ranked_signatures=ranked_survivors,
        score_rows=scores,
        extra_rows=extra_rows,
        top_n=min(3, len(ranked_survivors)),
    )

    final_report = {
        "task_family": "multi_hop_qa",
        "anchor_weights": dict(LEGACY_INHERITED_WEIGHTS),
        "search_radius": ANCHOR_SEARCH_RADIUS,
        "search_step": ANCHOR_SEARCH_STEP,
        "filter_stage": filter_result["method"],
        "selection_stage": "pending_pilot75",
        "winner_reason": None,
        "reject_reason": None if shortlist else "no_lodo_valid_candidates",
        "pilot75_metrics": {},
        "representative300_metrics": {},
        "inherited_baseline_metrics": {
            "hotpot_qa": summarize_candidate(inherited_hotpot),
            "two_wiki_multihop": summarize_candidate(inherited_twowiki),
        },
        "filter_result": filter_result,
        "candidate_count": len(grid),
        "survivor_count": len(filter_result["survivors"]),
        "shortlist": shortlist,
    }
    atomic_write_json(selection_path, final_report)


def run_open_qa(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    selection_path = out_dir / "open_qa_selection_report.json"
    candidates_path = out_dir / "open_qa_local_candidates.json"
    squad_rows = load_jsonl(args.squad_cache)
    controller_cfg = load_controller_cfg(args.controller_cfg_yaml)

    grid = build_anchor_local_grid(
        step=ANCHOR_SEARCH_STEP,
        anchor=LEGACY_INHERITED_WEIGHTS,
        radius=ANCHOR_SEARCH_RADIUS,
        bounds=OPEN_QA_LOCAL_BOUNDS,
        include=[LEGACY_INHERITED_WEIGHTS],
    )

    scores: dict[str, dict[str, Any]] = {}
    for weights in grid:
        sig = weight_signature(weights)
        eval_result = evaluate_controller_cache(squad_rows, task_family="open_qa", controller_cfg=controller_cfg, weights=weights)
        scores[sig] = {
            "weights": canonicalize_weights(weights),
            "summary": eval_result["summary"],
        }

    inherited_sig = weight_signature(LEGACY_INHERITED_WEIGHTS)
    inherited_summary = scores[inherited_sig]["summary"]
    export_rows: list[dict[str, Any]] = []
    extra_rows: dict[str, dict[str, Any]] = {}
    filtered_signatures: list[str] = []
    for sig, row in scores.items():
        passed = passes_open_qa_relative_gate(row["summary"], inherited_summary)
        if passed:
            filtered_signatures.append(sig)
        metrics = summarize_candidate(row["summary"])
        l1 = l1_distance_to_anchor(row["weights"], LEGACY_INHERITED_WEIGHTS)
        export_row = {
            "weight_signature": sig,
            "weights": dict(row["weights"]),
            "l1_distance_to_anchor": l1,
            "passes_filter": passed,
            "squad_open": {
                "metrics": metrics,
                "delta_vs_inherited": delta_vs_baseline(row["summary"], inherited_summary),
            },
        }
        export_rows.append(export_row)
        extra_rows[sig] = {
            "l1_distance_to_anchor": l1,
            "metrics": {"squad_open": metrics},
            "delta_vs_inherited": {"squad_open": delta_vs_baseline(row["summary"], inherited_summary)},
        }
    atomic_write_json(candidates_path, export_rows)

    ranked_filtered = sorted(
        [sig for sig in filtered_signatures if sig != inherited_sig],
        key=lambda sig: (
            anchor_distance_rank_key(scores[sig]["weights"], anchor=LEGACY_INHERITED_WEIGHTS),
            open_qa_rank_key(scores[sig]["summary"]),
        ),
    )
    shortlist = write_candidate_overrides(
        out_dir=out_dir,
        task_family="open_qa",
        ranked_signatures=ranked_filtered,
        score_rows=scores,
        extra_rows=extra_rows,
        top_n=min(5, len(ranked_filtered)),
    )

    final_report = {
        "task_family": "open_qa",
        "anchor_weights": dict(LEGACY_INHERITED_WEIGHTS),
        "search_radius": ANCHOR_SEARCH_RADIUS,
        "search_step": ANCHOR_SEARCH_STEP,
        "filter_stage": "relative_to_inherited_filter",
        "selection_stage": "pending_pilot75",
        "winner_reason": None,
        "reject_reason": None if shortlist else "no_relative_valid_candidates",
        "pilot75_metrics": {},
        "representative300_metrics": {},
        "inherited_baseline_metrics": summarize_candidate(inherited_summary),
        "candidate_count": len(grid),
        "survivor_count": len(filtered_signatures),
        "shortlist": shortlist,
    }
    atomic_write_json(selection_path, final_report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive task-family utility weights from offline controller caches.")
    sub = parser.add_subparsers(dest="mode", required=True)

    mh = sub.add_parser("multi_hop")
    mh.add_argument("--hotpot-cache", required=True)
    mh.add_argument("--twowiki-cache", required=True)
    mh.add_argument("--controller-cfg-yaml", required=True)
    mh.add_argument("--output-dir", required=True)

    oq = sub.add_parser("open_qa")
    oq.add_argument("--squad-cache", required=True)
    oq.add_argument("--controller-cfg-yaml", required=True)
    oq.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.mode == "multi_hop":
        run_multi_hop(args)
    else:
        run_open_qa(args)


if __name__ == "__main__":
    main()
