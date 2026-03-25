#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rag_cpu.benchmark_suite import run_qa_benchmark, write_summary_markdown
from rag_cpu.config import load_config, write_config
from rag_cpu.utils import save_json


def _load_stage2_rows(leaderboard_csv: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(leaderboard_csv.open("r", encoding="utf-8")))
    out = [r for r in rows if str(r.get("stage", "")).strip() == "stage2"]
    out.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    return out


def _config_from_stage2_manifest(stage2_dir: Path, cfg_id: str, base_cfg: dict[str, Any]) -> dict[str, Any]:
    manifest_path = stage2_dir / cfg_id / "run_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = payload.get("config")
        if isinstance(cfg, dict):
            return copy.deepcopy(cfg)
    return copy.deepcopy(base_cfg)


def _apply_sp3_p4(cfg: dict[str, Any]) -> None:
    runtime = cfg.setdefault("llm_runtime", {})
    runtime["sp3_enabled"] = True
    runtime["sp3_profile"] = "P4"


def _ensure_context_budgeting(cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
    if "context_budgeting" not in cfg or not isinstance(cfg.get("context_budgeting"), dict):
        cfg["context_budgeting"] = copy.deepcopy(base_cfg.get("context_budgeting", {}))


def _metric(summary: dict[str, Any], path: list[str], default: float = 0.0) -> float:
    cur: Any = summary
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    try:
        return float(cur)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired Stage-2 (static vs ACB) with SP3=P4")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--leaderboard", default="results/autotune_hotpot_v2/leaderboard.csv")
    parser.add_argument("--stage2-dir", default="results/autotune_hotpot_v2/stage2")
    parser.add_argument("--run-id", default="paired_stage2_p4_1000")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-ids-path", default="")
    parser.add_argument("--ui-update-every", type=int, default=10)
    parser.add_argument("--sampling-interval-ms", type=int, default=200)
    parser.add_argument("--timeseries-stride", type=int, default=5)
    parser.add_argument("--profile-timeseries", action="store_true")
    parser.add_argument("--profile-power", dest="profile_power", action="store_true")
    parser.add_argument("--no-profile-power", dest="profile_power", action="store_false")
    parser.set_defaults(profile_power=None)
    parser.add_argument("--power-sampling-interval-ms", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    leaderboard_path = Path(args.leaderboard)
    stage2_dir = Path(args.stage2_dir)
    run_root = Path(args.output_dir) / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    rows = _load_stage2_rows(leaderboard_path)
    if not rows:
        raise RuntimeError(f"No stage2 rows found in {leaderboard_path}")
    selected = rows[: max(1, int(args.top_k))]

    if args.query_ids_path:
        global_qids_path = Path(args.query_ids_path)
    else:
        top_id = selected[0]["config_id"]
        global_qids_path = stage2_dir / top_id / "sampled_qids.json"

    if not global_qids_path.exists():
        raise FileNotFoundError(f"query ids path not found: {global_qids_path}")

    metadata = {
        "run_id": args.run_id,
        "base_config": args.base_config,
        "leaderboard": str(leaderboard_path),
        "stage2_dir": str(stage2_dir),
        "top_k": int(args.top_k),
        "num_queries": int(args.num_queries),
        "seed": int(args.seed),
        "global_query_ids_path": str(global_qids_path),
        "selection_order_by": "stage2 score desc",
        "selected_config_ids": [r["config_id"] for r in selected],
        "sp3_profile": "P4",
        "variants": ["static_p4", "adaptive_p4_acb"],
    }
    save_json(run_root / "run_manifest.json", metadata)

    result_rows: list[dict[str, Any]] = []
    power_sampling_interval_ms = (
        int(args.power_sampling_interval_ms) if int(args.power_sampling_interval_ms) > 0 else None
    )

    for idx, row in enumerate(selected, start=1):
        cfg_id = row["config_id"]
        cfg_base = _config_from_stage2_manifest(stage2_dir=stage2_dir, cfg_id=cfg_id, base_cfg=base_cfg)
        _ensure_context_budgeting(cfg_base, base_cfg)

        # Variant A: static + SP3=P4
        cfg_static = copy.deepcopy(cfg_base)
        _apply_sp3_p4(cfg_static)
        cfg_static["context_budgeting"]["enabled"] = False

        # Variant B: adaptive + SP3=P4
        cfg_adaptive = copy.deepcopy(cfg_base)
        _apply_sp3_p4(cfg_adaptive)
        cfg_adaptive["context_budgeting"]["enabled"] = True

        variants = [
            ("static_p4", cfg_static),
            ("adaptive_p4_acb", cfg_adaptive),
        ]

        variant_summaries: dict[str, dict[str, Any]] = {}

        for variant_name, variant_cfg in variants:
            cfg_path = run_root / "configs" / f"{idx:02d}_{cfg_id}_{variant_name}.yaml"
            write_config(cfg_path, variant_cfg)

            out_dir = run_root / "runs" / f"{idx:02d}_{cfg_id}" / variant_name / "hotpot_qa"
            summary_path = out_dir / "summary.json"

            if args.resume and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = run_qa_benchmark(
                    cfg=variant_cfg,
                    dataset="hotpot_qa",
                    tier="B",
                    output_dir=out_dir,
                    run_id=args.run_id,
                    seed=int(args.seed),
                    max_queries_override=int(args.num_queries),
                    retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                    profile_timeseries=bool(args.profile_timeseries),
                    sampling_interval_ms=int(args.sampling_interval_ms),
                    timeseries_stride=int(args.timeseries_stride),
                    profile_power=args.profile_power,
                    power_sampling_interval_ms=power_sampling_interval_ms,
                    ui_update_every=int(args.ui_update_every),
                    query_ids_path=global_qids_path,
                )
                save_json(summary_path, summary)
                write_summary_markdown(out_dir / "summary.md", summary)

            variant_summaries[variant_name] = summary

        static_s = variant_summaries["static_p4"]
        adapt_s = variant_summaries["adaptive_p4_acb"]

        qids_static = json.loads((run_root / "runs" / f"{idx:02d}_{cfg_id}" / "static_p4" / "hotpot_qa" / "sampled_qids.json").read_text(encoding="utf-8"))
        qids_adapt = json.loads((run_root / "runs" / f"{idx:02d}_{cfg_id}" / "adaptive_p4_acb" / "hotpot_qa" / "sampled_qids.json").read_text(encoding="utf-8"))
        qids_identical = qids_static == qids_adapt

        out_row = {
            "rank": idx,
            "config_id": cfg_id,
            "qids_identical": bool(qids_identical),
            "static_f1": _metric(static_s, ["generation", "F1"]),
            "adaptive_f1": _metric(adapt_s, ["generation", "F1"]),
            "delta_f1": _metric(adapt_s, ["generation", "F1"]) - _metric(static_s, ["generation", "F1"]),
            "static_em": _metric(static_s, ["generation", "EM"]),
            "adaptive_em": _metric(adapt_s, ["generation", "EM"]),
            "delta_em": _metric(adapt_s, ["generation", "EM"]) - _metric(static_s, ["generation", "EM"]),
            "static_ttft_p95_ms": _metric(static_s, ["latency_ms", "ttft_ms", "p95"]),
            "adaptive_ttft_p95_ms": _metric(adapt_s, ["latency_ms", "ttft_ms", "p95"]),
            "delta_ttft_p95_ms": _metric(adapt_s, ["latency_ms", "ttft_ms", "p95"]) - _metric(static_s, ["latency_ms", "ttft_ms", "p95"]),
            "static_t_total_p95_ms": _metric(static_s, ["latency_ms", "t_total_ms", "p95"]),
            "adaptive_t_total_p95_ms": _metric(adapt_s, ["latency_ms", "t_total_ms", "p95"]),
            "delta_t_total_p95_ms": _metric(adapt_s, ["latency_ms", "t_total_ms", "p95"]) - _metric(static_s, ["latency_ms", "t_total_ms", "p95"]),
            "static_failure_rate": _metric(static_s, ["generation", "failure_rate"]),
            "adaptive_failure_rate": _metric(adapt_s, ["generation", "failure_rate"]),
            "delta_failure_rate": _metric(adapt_s, ["generation", "failure_rate"]) - _metric(static_s, ["generation", "failure_rate"]),
            "static_context_used_p95": _metric(static_s, ["context_budgeting", "context_tokens_used", "p95"]),
            "adaptive_context_used_p95": _metric(adapt_s, ["context_budgeting", "context_tokens_used", "p95"]),
            "delta_context_used_p95": _metric(adapt_s, ["context_budgeting", "context_tokens_used", "p95"]) - _metric(static_s, ["context_budgeting", "context_tokens_used", "p95"]),
            "static_summary_path": str(run_root / "runs" / f"{idx:02d}_{cfg_id}" / "static_p4" / "hotpot_qa" / "summary.json"),
            "adaptive_summary_path": str(run_root / "runs" / f"{idx:02d}_{cfg_id}" / "adaptive_p4_acb" / "hotpot_qa" / "summary.json"),
        }
        result_rows.append(out_row)
        save_json(run_root / "paired_stage2_progress.json", {"rows": result_rows})

    save_json(run_root / "paired_stage2_summary.json", {"rows": result_rows, "metadata": metadata})

    fieldnames = list(result_rows[0].keys()) if result_rows else []
    if fieldnames:
        with (run_root / "paired_stage2_summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in result_rows:
                w.writerow(r)

    md_lines = [
        "# Paired Stage-2 (SP3=P4)",
        "",
        f"- run_id: `{args.run_id}`",
        f"- top_k: `{args.top_k}`",
        f"- num_queries: `{args.num_queries}`",
        f"- global_query_ids_path: `{global_qids_path}`",
        "",
        "| Rank | Config | qids_identical | ΔF1 | ΔEM | ΔTTFT p95 (ms) | Δt_total p95 (ms) | Δfailure_rate | Δcontext_used p95 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result_rows:
        md_lines.append(
            "| {rank} | {config_id} | {qids_identical} | {delta_f1:+.4f} | {delta_em:+.4f} | {delta_ttft_p95_ms:+.1f} | {delta_t_total_p95_ms:+.1f} | {delta_failure_rate:+.4f} | {delta_context_used_p95:+.1f} |".format(
                **r
            )
        )
    (run_root / "paired_stage2_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Done. run_id={args.run_id}")
    print(f"Output: {run_root}")


if __name__ == "__main__":
    main()
