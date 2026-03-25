#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rag_cpu.benchmark_suite import run_qa_benchmark, write_summary_markdown
from rag_cpu.config import deep_update, load_config, write_config
from rag_cpu.utils import save_json


CONFIG_SPECS = [
    ("C0_a1_baseline", "configs/pool_v1/a1_baseline.yaml"),
    ("C1_a1_v1_postproc_conservative", "configs/pool_v1/a1_v1_postproc_conservative.yaml"),
    ("C2_a1_v2_retrieval_recall", "configs/pool_v1/a1_v2_retrieval_recall.yaml"),
    ("C3_a1_v3_context_denoise", "configs/pool_v1/a1_v3_context_denoise.yaml"),
    ("C4_a1_v4_combo", "configs/pool_v1/a1_v4_combo.yaml"),
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return out


def _metric(payload: dict[str, Any], path: list[str], default: float = 0.0) -> float:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return float(default)
        cur = cur.get(key)
    return _safe_float(cur, default)


def _extract_f1(row: dict[str, Any]) -> float:
    metrics = row.get("answer_metrics_per_query", {})
    if isinstance(metrics, dict):
        f1 = _safe_float(metrics.get("f1", None), default=-1.0)
        if f1 >= 0.0:
            return f1
    return 0.0


def _compute_strata_f1(per_query_path: Path, strata_by_qid: dict[str, str]) -> dict[str, float]:
    sums = {"hard": 0.0, "mid": 0.0, "easy": 0.0}
    counts = {"hard": 0, "mid": 0, "easy": 0}

    with per_query_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("qid", "")).strip()
            if not qid or qid not in strata_by_qid:
                continue
            stratum = strata_by_qid[qid]
            if stratum not in sums:
                continue
            sums[stratum] += _extract_f1(row)
            counts[stratum] += 1

    out: dict[str, float] = {}
    for key in ("hard", "mid", "easy"):
        out[key] = float(sums[key] / counts[key]) if counts[key] > 0 else 0.0
    out["counts"] = counts
    return out


def _run_single(
    cfg: dict[str, Any],
    run_id: str,
    out_dir: Path,
    seed: int,
    num_queries: int,
    query_ids_path: Path,
    ui_update_every: int,
    profile_timeseries: bool,
    sampling_interval_ms: int,
    timeseries_stride: int,
    profile_power: bool | None,
    power_sampling_interval_ms: int | None,
    resume: bool,
) -> dict[str, Any]:
    summary_path = out_dir / "summary.json"
    if resume and summary_path.exists():
        return _load_json(summary_path)

    summary = run_qa_benchmark(
        cfg=cfg,
        dataset="hotpot_qa",
        tier="B",
        output_dir=out_dir,
        run_id=run_id,
        seed=int(seed),
        max_queries_override=int(num_queries),
        retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
        profile_timeseries=bool(profile_timeseries),
        sampling_interval_ms=int(sampling_interval_ms),
        timeseries_stride=int(timeseries_stride),
        profile_power=profile_power,
        power_sampling_interval_ms=power_sampling_interval_ms,
        ui_update_every=int(ui_update_every),
        query_ids_path=query_ids_path,
    )
    save_json(summary_path, summary)
    write_summary_markdown(out_dir / "summary.md", summary)
    return summary


def _select_top2(pool_rows: list[dict[str, Any]], guardrail_factor: float = 1.35) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_name = {r["config_name"]: r for r in pool_rows}
    baseline = by_name.get("C0_a1_baseline")
    if baseline is None:
        raise RuntimeError("baseline C0_a1_baseline missing from pool rows")

    b_ttotal = baseline["t_total_p95_ms"]
    b_ttft = baseline["ttft_p95_ms"]
    ttotal_th = b_ttotal * guardrail_factor if b_ttotal > 0 else float("inf")
    ttft_th = b_ttft * guardrail_factor if b_ttft > 0 else float("inf")

    for row in pool_rows:
        row["guardrail_pass"] = bool(
            row["t_total_p95_ms"] <= ttotal_th and row["ttft_p95_ms"] <= ttft_th
        )

    guardrail_pass_rows = [r for r in pool_rows if r["guardrail_pass"]]
    ranking_key = lambda r: (
        -r["F1_pool"],
        -r["EM_pool"],
        -r["F1_weighted"],
        r["t_total_p95_ms"],
    )

    guardrail_failed = False
    source = guardrail_pass_rows
    if not source:
        guardrail_failed = True
        source = pool_rows

    top2 = sorted(source, key=ranking_key)[:2]
    meta = {
        "guardrail_factor": float(guardrail_factor),
        "baseline_t_total_p95_ms": float(b_ttotal),
        "baseline_ttft_p95_ms": float(b_ttft),
        "threshold_t_total_p95_ms": float(ttotal_th),
        "threshold_ttft_p95_ms": float(ttft_th),
        "guardrail_failed": bool(guardrail_failed),
    }
    return top2, meta


def _winner_stage2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda r: (-r["F1_1000"], -r["EM_1000"], r["t_total_p95_ms"]),
    )[0]


def _write_selection_report(
    out_path: Path,
    pool_rows: list[dict[str, Any]],
    selection_meta: dict[str, Any],
    top2: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    winner: dict[str, Any],
) -> None:
    by_name = {r["config_name"]: r for r in pool_rows}
    baseline = by_name["C0_a1_baseline"]

    lines: list[str] = []
    lines.append("# Selection Report - Pool v1")
    lines.append("")
    lines.append(f"- guardrail_factor: {selection_meta['guardrail_factor']:.2f}")
    lines.append(f"- baseline_t_total_p95_ms: {selection_meta['baseline_t_total_p95_ms']:.2f}")
    lines.append(f"- baseline_ttft_p95_ms: {selection_meta['baseline_ttft_p95_ms']:.2f}")
    lines.append(f"- threshold_t_total_p95_ms: {selection_meta['threshold_t_total_p95_ms']:.2f}")
    lines.append(f"- threshold_ttft_p95_ms: {selection_meta['threshold_ttft_p95_ms']:.2f}")
    lines.append(f"- guardrail_failed: {selection_meta['guardrail_failed']}")
    lines.append("")

    lines.append("## Pool 300 Scores")
    lines.append("")
    lines.append(
        "| Config | F1_pool | EM_pool | F1_hard | F1_mid | F1_easy | F1_weighted | t_total_p95_ms | ttft_p95_ms | guardrail_pass | ΔF1 vs C0 | ΔEM vs C0 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in sorted(pool_rows, key=lambda r: r["config_name"]):
        d_f1 = row["F1_pool"] - baseline["F1_pool"]
        d_em = row["EM_pool"] - baseline["EM_pool"]
        lines.append(
            "| {config_name} | {F1_pool:.4f} | {EM_pool:.4f} | {F1_hard:.4f} | {F1_mid:.4f} | {F1_easy:.4f} | {F1_weighted:.4f} | {t_total_p95_ms:.2f} | {ttft_p95_ms:.2f} | {guardrail_pass} | {d_f1:+.4f} | {d_em:+.4f} |".format(
                d_f1=d_f1,
                d_em=d_em,
                **row,
            )
        )

    lines.append("")
    lines.append("## Top-2 Selected")
    lines.append("")
    for i, row in enumerate(top2, start=1):
        lines.append(f"{i}. `{row['config_name']}` - F1={row['F1_pool']:.4f}, EM={row['EM_pool']:.4f}, F1_weighted={row['F1_weighted']:.4f}")

    if stage2_rows:
        lines.append("")
        lines.append("## Stage2 1000 Confirmation")
        lines.append("")
        lines.append("| Config | F1_1000 | EM_1000 | t_total_p95_ms | ttft_p95_ms |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in stage2_rows:
            lines.append(
                "| {config_name} | {F1_1000:.4f} | {EM_1000:.4f} | {t_total_p95_ms:.2f} | {ttft_p95_ms:.2f} |".format(
                    **row
                )
            )
        if winner:
            lines.append("")
            lines.append(
                f"Winner: `{winner['config_name']}` (F1_1000={winner['F1_1000']:.4f}, EM_1000={winner['EM_1000']:.4f}, t_total_p95_ms={winner['t_total_p95_ms']:.2f})"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool v1 matrix runner (A1 baseline + 4 variants)")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--pool-qids", default="qids/hotpot_repr300_v1.json")
    parser.add_argument("--pool-report", default="results/pool_repr300_v1_report.json")
    parser.add_argument("--stage2-qids", default="qids/hotpot_stage2_1000.json")
    parser.add_argument("--output-dir", default="results/pool_v1")
    parser.add_argument("--run-id", default="pool_v1_matrix")
    parser.add_argument("--pool-num-queries", type=int, default=300)
    parser.add_argument("--confirm-num-queries", type=int, default=1000)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ui-update-every", type=int, default=10)
    parser.add_argument("--profile-timeseries", action="store_true")
    parser.add_argument("--sampling-interval-ms", type=int, default=200)
    parser.add_argument("--timeseries-stride", type=int, default=5)
    parser.add_argument("--profile-power", dest="profile_power", action="store_true")
    parser.add_argument("--no-profile-power", dest="profile_power", action="store_false")
    parser.set_defaults(profile_power=None)
    parser.add_argument("--power-sampling-interval-ms", type=int, default=0)
    parser.add_argument("--stage", choices=["all", "pool", "confirm"], default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pool_qids_path = Path(args.pool_qids)
    if not pool_qids_path.exists():
        raise FileNotFoundError(f"pool qids not found: {pool_qids_path}")
    pool_qids = _load_json(pool_qids_path)
    if not isinstance(pool_qids, list) or len(pool_qids) < int(args.pool_num_queries):
        raise RuntimeError(
            f"pool_qids must contain at least {int(args.pool_num_queries)} ids: {pool_qids_path}"
        )

    pool_report_path = Path(args.pool_report)
    if not pool_report_path.exists():
        raise FileNotFoundError(f"pool report not found: {pool_report_path}")
    pool_report = _load_json(pool_report_path)
    selected_meta = pool_report.get("selected_metadata", [])
    if not isinstance(selected_meta, list) or not selected_meta:
        raise RuntimeError("pool report missing selected_metadata")
    strata_by_qid = {
        str(row.get("qid", "")): str(row.get("stratum", ""))
        for row in selected_meta
        if str(row.get("qid", "")).strip()
    }

    stage2_qids_path = Path(args.stage2_qids)
    if args.stage in {"all", "confirm"} and not stage2_qids_path.exists():
        raise FileNotFoundError(f"stage2 qids not found: {stage2_qids_path}")

    base_cfg = load_config(args.base_config)
    power_sampling_interval_ms = (
        int(args.power_sampling_interval_ms) if int(args.power_sampling_interval_ms) > 0 else None
    )

    (out_root / "configs").mkdir(parents=True, exist_ok=True)
    (out_root / "runs" / "pool").mkdir(parents=True, exist_ok=True)
    (out_root / "runs" / "stage2").mkdir(parents=True, exist_ok=True)

    pool_rows: list[dict[str, Any]] = []
    leaderboard_path = out_root / "leaderboard_pool300.json"

    if args.stage in {"all", "pool"}:
        config_specs = list(CONFIG_SPECS)
        if int(args.max_configs) > 0:
            config_specs = config_specs[: int(args.max_configs)]

        for config_name, override_path in config_specs:
            override_cfg = load_config(override_path)
            cfg = deep_update(base_cfg, override_cfg)
            cfg_out = out_root / "configs" / f"{config_name}.yaml"
            write_config(cfg_out, cfg)

            run_dir = out_root / "runs" / "pool" / config_name / "hotpot_qa"
            summary = _run_single(
                cfg=cfg,
                run_id=args.run_id,
                out_dir=run_dir,
                seed=int(args.seed),
                num_queries=int(args.pool_num_queries),
                query_ids_path=pool_qids_path,
                ui_update_every=int(args.ui_update_every),
                profile_timeseries=bool(args.profile_timeseries),
                sampling_interval_ms=int(args.sampling_interval_ms),
                timeseries_stride=int(args.timeseries_stride),
                profile_power=args.profile_power,
                power_sampling_interval_ms=power_sampling_interval_ms,
                resume=bool(args.resume),
            )
            strata = _compute_strata_f1(run_dir / "per_query.jsonl", strata_by_qid)
            f1_hard = _safe_float(strata.get("hard", 0.0))
            f1_mid = _safe_float(strata.get("mid", 0.0))
            f1_easy = _safe_float(strata.get("easy", 0.0))
            row = {
                "config_name": config_name,
                "override_path": override_path,
                "effective_config": str(cfg_out),
                "summary_path": str(run_dir / "summary.json"),
                "per_query_path": str(run_dir / "per_query.jsonl"),
                "F1_pool": _metric(summary, ["generation", "F1"]),
                "EM_pool": _metric(summary, ["generation", "EM"]),
                "F1_hard": f1_hard,
                "F1_mid": f1_mid,
                "F1_easy": f1_easy,
                "F1_weighted": float(0.5 * f1_hard + 0.3 * f1_mid + 0.2 * f1_easy),
                "t_total_p95_ms": _metric(summary, ["latency_ms", "t_total_ms", "p95"]),
                "ttft_p95_ms": _metric(summary, ["latency_ms", "ttft_ms", "p95"]),
                "failure_rate": _metric(summary, ["generation", "failure_rate"]),
                "valid_answer_rate": _metric(summary, ["generation", "valid_answer_rate"]),
                "strata_counts": strata.get("counts", {}),
            }
            pool_rows.append(row)

        top2, selection_meta = _select_top2(pool_rows)
        payload = {
            "task": "pool_v1_matrix",
            "run_id": args.run_id,
            "seed": int(args.seed),
            "pool_qids_path": str(pool_qids_path),
            "pool_report_path": str(pool_report_path),
            "pool_size": int(len(pool_qids)),
            "pool_num_queries": int(args.pool_num_queries),
            "max_configs": int(args.max_configs),
            "pool_results": pool_rows,
            "selection": {
                **selection_meta,
                "top2": [r["config_name"] for r in top2],
            },
        }
        save_json(leaderboard_path, payload)
        stage2_rows: list[dict[str, Any]] = []
        winner: dict[str, Any] = {}
        _write_selection_report(
            out_path=out_root / "selection_report.md",
            pool_rows=pool_rows,
            selection_meta=selection_meta,
            top2=top2,
            stage2_rows=stage2_rows,
            winner=winner,
        )

    if args.stage in {"all", "confirm"}:
        if leaderboard_path.exists():
            payload = _load_json(leaderboard_path)
            pool_rows = payload.get("pool_results", [])
            top2_names = payload.get("selection", {}).get("top2", [])
            if not top2_names:
                top2, selection_meta = _select_top2(pool_rows)
                top2_names = [r["config_name"] for r in top2]
                payload["selection"] = {**selection_meta, "top2": top2_names}
        else:
            raise RuntimeError("leaderboard_pool300.json not found; run --stage pool first")

        by_name = {name: path for name, path in CONFIG_SPECS}
        stage2_rows: list[dict[str, Any]] = []
        for config_name in top2_names[:2]:
            if config_name not in by_name:
                raise RuntimeError(f"config not found in pool matrix: {config_name}")
            override_cfg = load_config(by_name[config_name])
            cfg = deep_update(base_cfg, override_cfg)
            cfg_out = out_root / "configs" / f"{config_name}_stage2.yaml"
            write_config(cfg_out, cfg)
            run_dir = out_root / "runs" / "stage2" / config_name / "hotpot_qa"
            summary = _run_single(
                cfg=cfg,
                run_id=f"{args.run_id}_stage2",
                out_dir=run_dir,
                seed=int(args.seed),
                num_queries=int(args.confirm_num_queries),
                query_ids_path=stage2_qids_path,
                ui_update_every=int(args.ui_update_every),
                profile_timeseries=bool(args.profile_timeseries),
                sampling_interval_ms=int(args.sampling_interval_ms),
                timeseries_stride=int(args.timeseries_stride),
                profile_power=args.profile_power,
                power_sampling_interval_ms=power_sampling_interval_ms,
                resume=bool(args.resume),
            )
            stage2_rows.append(
                {
                    "config_name": config_name,
                    "summary_path": str(run_dir / "summary.json"),
                    "F1_1000": _metric(summary, ["generation", "F1"]),
                    "EM_1000": _metric(summary, ["generation", "EM"]),
                    "t_total_p95_ms": _metric(summary, ["latency_ms", "t_total_ms", "p95"]),
                    "ttft_p95_ms": _metric(summary, ["latency_ms", "ttft_ms", "p95"]),
                    "failure_rate": _metric(summary, ["generation", "failure_rate"]),
                }
            )

        winner = _winner_stage2(stage2_rows)
        payload["stage2_confirm"] = {
            "qids_path": str(stage2_qids_path),
            "num_queries": 1000,
            "num_queries_effective": int(args.confirm_num_queries),
            "results": stage2_rows,
            "winner": winner,
            "winner_rule": "max F1_1000, tie EM_1000, tie min t_total_p95_ms",
        }
        save_json(leaderboard_path, payload)

        top2_rows = [r for r in pool_rows if r.get("config_name") in set(payload.get("selection", {}).get("top2", []))]
        _write_selection_report(
            out_path=out_root / "selection_report.md",
            pool_rows=pool_rows,
            selection_meta=payload.get("selection", {}),
            top2=top2_rows,
            stage2_rows=stage2_rows,
            winner=winner,
        )

    print(f"Done. Artifacts under: {out_root}")


if __name__ == "__main__":
    main()
