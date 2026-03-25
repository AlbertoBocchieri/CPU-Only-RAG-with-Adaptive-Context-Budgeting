#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rag_cpu.benchmark_suite import (
    config_fingerprint,
    run_beir_retrieval_benchmark,
    run_qa_benchmark,
    write_summary_markdown,
)
from rag_cpu.config import load_config, write_config
from rag_cpu.utils import now_ts, save_json


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "run_id": "",
            "rows": [],
            "stage1_candidates": [],
            "stage1_survivors": [],
            "stage2_survivors": [],
            "best_quality_config_id": "",
            "best_efficiency_config_id": "",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _composite_score(f1: float, ttft_ms: float, rss_peak_mb: float, w_ttft: float, w_rss: float) -> float:
    import math

    return float(f1 - w_ttft * math.log1p(max(0.0, ttft_ms)) - w_rss * math.log1p(max(0.0, rss_peak_mb)))


def _is_dominated(a: dict[str, Any], b: dict[str, Any]) -> bool:
    better_or_equal = (
        b["f1"] >= a["f1"]
        and b["ttft_ms"] <= a["ttft_ms"]
        and b["rss_peak_mb"] <= a["rss_peak_mb"]
    )
    strictly_better = (
        b["f1"] > a["f1"]
        or b["ttft_ms"] < a["ttft_ms"]
        or b["rss_peak_mb"] < a["rss_peak_mb"]
    )
    return bool(better_or_equal and strictly_better)


def _mark_pareto(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        r["pareto_optimal"] = True
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if i == j:
                continue
            if _is_dominated(a, b):
                a["pareto_optimal"] = False
                break


def _write_leaderboard(run_root: Path, rows: list[dict[str, Any]]) -> None:
    save_json(run_root / "leaderboard.json", rows)
    fields = [
        "stage",
        "tier",
        "dataset",
        "config_id",
        "retriever_mode",
        "fusion_method",
        "k",
        "reranker_enabled",
        "rerank_top_in",
        "context_packing",
        "prompt_variant",
        "max_tokens",
        "f1",
        "em",
        "ttft_ms",
        "rss_peak_mb",
        "score",
        "pareto_optimal",
        "summary_path",
    ]
    with (run_root / "leaderboard.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _make_candidate_grid(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    all_candidates: dict[str, list[dict[str, Any]]] = {"hybrid": [], "dense_only": []}
    for mode in ["hybrid", "dense_only"]:
        for k in [5, 10, 20]:
            for reranker_enabled in [False, True]:
                rerank_options = [20, 50] if reranker_enabled else [20]
                for rerank_top_in in rerank_options:
                    for context_packing in [False, True]:
                        for prompt_variant in ["evidence_only", "multihop_short_reasoning"]:
                            for max_tokens in [64, 128]:
                                all_candidates[mode].append(
                                    {
                                        "retriever_mode": mode,
                                        "k": k,
                                        "reranker_enabled": reranker_enabled,
                                        "rerank_top_in": rerank_top_in,
                                        "context_packing": context_packing,
                                        "prompt_variant": prompt_variant,
                                        "max_tokens": max_tokens,
                                        "fusion_method": "RRF",
                                        "rrf_k": 60,
                                    }
                                )

    rng.shuffle(all_candidates["hybrid"])
    rng.shuffle(all_candidates["dense_only"])
    selected = all_candidates["hybrid"][:16] + all_candidates["dense_only"][:8]
    rng.shuffle(selected)
    return selected


def _apply_prompt_variant(cfg: dict[str, Any], variant: str) -> None:
    if variant == "multihop_short_reasoning":
        cfg["llm"]["prompt_mode"] = "direct"
        cfg["llm"]["direct_prompt_template"] = (
            "Context:\n"
            "{context}\n\n"
            "Answer the multi-hop question with a concise factual answer grounded in the context.\n"
            "If evidence is insufficient, reply: Non lo so.\n\n"
            "Q: {question}\n"
            "A:"
        )
    else:
        cfg["llm"]["prompt_mode"] = "direct"
        cfg["llm"]["direct_prompt_template"] = (
            "Context:\n"
            "{context}\n\n"
            "Based on the context above, answer this question with just the answer, no explanation:\n\n"
            "Q: {question}\n"
            "A:"
        )


def _config_from_candidate(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["retrieval"]["retriever_mode"] = cand["retriever_mode"]
    cfg["retrieval"]["fusion_method"] = cand["fusion_method"]
    cfg["retrieval"]["rrf_k"] = int(cand["rrf_k"])
    cfg["retrieval"]["top_k_final"] = int(cand["k"])
    cfg["reranker"]["enabled"] = bool(cand["reranker_enabled"])
    cfg["reranker"]["top_k_in"] = int(cand["rerank_top_in"])
    cfg["reranker"]["top_k_out"] = int(min(cand["k"], 10))
    cfg["llm"]["context_packing"] = bool(cand["context_packing"])
    cfg["llm"]["max_new_tokens"] = int(cand["max_tokens"])
    _apply_prompt_variant(cfg, cand["prompt_variant"])
    return cfg


def _row_from_summary(
    summary: dict[str, Any],
    candidate: dict[str, Any],
    stage: str,
    tier: str,
    dataset: str,
    w_ttft: float,
    w_rss: float,
    summary_path: str,
) -> dict[str, Any]:
    f1 = float(summary.get("generation", {}).get("F1", 0.0))
    em = float(summary.get("generation", {}).get("EM", 0.0))
    ttft_ms = float(summary.get("latency_ms", {}).get("ttft_ms", {}).get("mean", 0.0))
    rss_peak_mb = float(summary.get("resources", {}).get("rss_peak_bytes", {}).get("mean", 0.0)) / (1024 * 1024)
    return {
        "stage": stage,
        "tier": tier,
        "dataset": dataset,
        "config_id": summary.get("config_id", ""),
        "retriever_mode": candidate.get("retriever_mode", ""),
        "fusion_method": candidate.get("fusion_method", ""),
        "k": int(candidate.get("k", 0)),
        "reranker_enabled": bool(candidate.get("reranker_enabled", False)),
        "rerank_top_in": int(candidate.get("rerank_top_in", 0)),
        "context_packing": bool(candidate.get("context_packing", False)),
        "prompt_variant": candidate.get("prompt_variant", ""),
        "max_tokens": int(candidate.get("max_tokens", 0)),
        "f1": f1,
        "em": em,
        "ttft_ms": ttft_ms,
        "rss_peak_mb": rss_peak_mb,
        "score": _composite_score(f1, ttft_ms, rss_peak_mb, w_ttft=w_ttft, w_rss=w_rss),
        "pareto_optimal": False,
        "summary_path": summary_path,
    }


def _pick_survivors(rows: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    _mark_pareto(rows)
    pareto = sorted([r for r in rows if r.get("pareto_optimal")], key=lambda x: x["score"], reverse=True)
    selected = pareto[:keep]
    if len(selected) < keep:
        rest = sorted([r for r in rows if r not in selected], key=lambda x: x["score"], reverse=True)
        selected.extend(rest[: keep - len(selected)])
    return selected


def _find_row(rows: list[dict[str, Any]], stage: str, config_id: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("stage") == stage and row.get("config_id") == config_id:
            return row
    return None


def _bm25_baselines(base_cfg: dict[str, Any], k_values: list[int]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for k in k_values:
        cfg = copy.deepcopy(base_cfg)
        cfg["retrieval"]["retriever_mode"] = "bm25_only"
        cfg["retrieval"]["fusion_method"] = "RRF"
        cfg["retrieval"]["rrf_k"] = 60
        cfg["retrieval"]["top_k_final"] = int(k)
        cfg["reranker"]["enabled"] = False
        meta = {
            "retriever_mode": "bm25_only",
            "fusion_method": "RRF",
            "k": int(k),
            "reranker_enabled": False,
            "rerank_top_in": 0,
            "context_packing": False,
            "prompt_variant": "evidence_only",
            "max_tokens": int(cfg["llm"]["max_new_tokens"]),
        }
        out.append((cfg, meta))
    return out


def _with_sp3_profile(cfg: dict[str, Any], profile: str) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    runtime = out.setdefault("llm_runtime", {})
    runtime["sp3_enabled"] = True
    runtime["sp3_profile"] = str(profile).upper()
    return out


def _pick_sp3_global_winner(
    run_root: Path,
    state_path: Path,
    state: dict[str, Any],
    ref_cfg: dict[str, Any],
    run_id: str,
    seed: int,
    pretest_queries: int,
    w_ttft: float,
    w_rss: float,
    sampling_interval_ms: int,
    timeseries_stride: int,
    profile_power: bool | None,
    power_sampling_interval_ms: int | None,
    ui_update_every: int,
) -> str:
    existing = str(state.get("sp3_global_winner", "")).upper().strip()
    if existing in {"P4", "P4B6", "P6B6"}:
        return existing

    profiles = ["P4", "P4B6", "P6B6"]
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        cfg = _with_sp3_profile(ref_cfg, profile)
        out_dir = run_root / "sp3_global_pretest" / profile
        summary = run_qa_benchmark(
            cfg=cfg,
            dataset="hotpot_qa",
            tier="A",
            output_dir=out_dir,
            run_id=run_id,
            seed=int(seed),
            max_queries_override=int(pretest_queries),
            retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
            profile_timeseries=False,
            sampling_interval_ms=int(sampling_interval_ms),
            timeseries_stride=int(timeseries_stride),
            profile_power=profile_power,
            power_sampling_interval_ms=power_sampling_interval_ms,
            ui_update_every=int(ui_update_every),
        )
        write_summary_markdown(out_dir / "summary.md", summary)
        generation = summary.get("generation", {})
        latency = summary.get("latency_ms", {})
        ttft = latency.get("ttft_ms", {})
        t_total = latency.get("t_total_ms", {})
        f1 = float(summary.get("generation", {}).get("F1", 0.0))
        em = float(generation.get("EM", 0.0))
        failure_rate = float(generation.get("failure_rate", 0.0))
        ttft_ms = float(ttft.get("mean", 0.0))
        ttft_p95 = float(ttft.get("p95", 0.0))
        t_total_p95 = float(t_total.get("p95", 0.0))
        rss_peak_mb = float(summary.get("resources", {}).get("rss_peak_bytes", {}).get("mean", 0.0)) / (1024 * 1024)
        score = _composite_score(f1=f1, ttft_ms=ttft_ms, rss_peak_mb=rss_peak_mb, w_ttft=w_ttft, w_rss=w_rss)
        rows.append(
            {
                "profile": profile,
                "em": em,
                "f1": f1,
                "failure_rate": failure_rate,
                "ttft_ms": ttft_ms,
                "ttft_p95_ms": ttft_p95,
                "t_total_p95_ms": t_total_p95,
                "rss_peak_mb": rss_peak_mb,
                "score": score,
                "summary_path": str(out_dir / "summary.json"),
            }
        )

    # SP3 selection priority for CPU-edge:
    # 1) minimize total latency p95
    # 2) tie-break: minimize TTFT p95
    # 3) tie-break: minimize failure_rate
    # 4) tie-break: maximize F1
    rows.sort(
        key=lambda x: (
            float(x.get("t_total_p95_ms", 0.0)),
            float(x.get("ttft_p95_ms", 0.0)),
            float(x.get("failure_rate", 0.0)),
            -float(x.get("f1", 0.0)),
        )
    )
    winner = rows[0]["profile"]
    state["sp3_global_pretest_rows"] = rows
    state["sp3_global_winner"] = winner
    _save_state(state_path, state)
    save_json(
        run_root / "sp3_global_pretest.json",
        {
            "winner": winner,
            "selection_criteria": [
                "min(t_total_p95_ms)",
                "min(ttft_p95_ms)",
                "min(failure_rate)",
                "max(f1)",
            ],
            "rows": rows,
        },
    )
    return winner


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable autotuning with successive halving")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--stage", default="1,2,3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--w-ttft", type=float, default=0.02)
    parser.add_argument("--w-rss", type=float, default=0.01)
    parser.add_argument("--profile-timeseries", action="store_true")
    parser.add_argument("--sampling-interval-ms", type=int, default=200)
    parser.add_argument("--timeseries-stride", type=int, default=5)
    parser.add_argument("--profile-power", dest="profile_power", action="store_true")
    parser.add_argument("--no-profile-power", dest="profile_power", action="store_false")
    parser.set_defaults(profile_power=None)
    parser.add_argument("--power-sampling-interval-ms", type=int, default=0)
    parser.add_argument("--ui-update-every", type=int, default=5)
    parser.add_argument("--run-alpha-sweep", action="store_true")
    parser.add_argument("--stage1-queries", type=int, default=200)
    parser.add_argument("--stage2-queries", type=int, default=1000)
    parser.add_argument("--stage1-max-configs", type=int, default=0)
    parser.add_argument("--stage2-max-configs", type=int, default=0)
    parser.add_argument("--skip-bm25-ref", action="store_true")
    parser.add_argument("--sp3-global-pretest", action="store_true")
    parser.add_argument("--sp3-pretest-queries", type=int, default=50)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    run_id = args.run_id or f"autotune_{now_ts()}"
    run_root = Path(args.output_dir) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "autotune_state.json"
    state = _load_state(state_path) if args.resume else _load_state(Path("__missing__"))
    if not state.get("run_id"):
        state["run_id"] = run_id

    requested_stages = {x.strip() for x in args.stage.split(",") if x.strip()}
    power_sampling_interval_ms = (
        int(args.power_sampling_interval_ms) if int(args.power_sampling_interval_ms) > 0 else None
    )
    rows: list[dict[str, Any]] = list(state.get("rows", []))
    candidates: list[dict[str, Any]] = state.get("stage1_candidates", [])
    if not candidates:
        candidates = _make_candidate_grid(seed=int(args.seed))
        state["stage1_candidates"] = candidates

    run_sp3_pretest = bool(args.sp3_global_pretest or base_cfg.get("llm_runtime", {}).get("sp3_enabled", False))
    if run_sp3_pretest:
        ref_candidate = candidates[0] if candidates else {
            "retriever_mode": "hybrid",
            "k": int(base_cfg["retrieval"]["top_k_final"]),
            "reranker_enabled": bool(base_cfg["reranker"]["enabled"]),
            "rerank_top_in": int(base_cfg["reranker"]["top_k_in"]),
            "context_packing": bool(base_cfg["llm"].get("context_packing", False)),
            "prompt_variant": "evidence_only",
            "max_tokens": int(base_cfg["llm"]["max_new_tokens"]),
            "fusion_method": "RRF",
            "rrf_k": 60,
        }
        ref_cfg = _config_from_candidate(base_cfg, ref_candidate)
        winner_profile = _pick_sp3_global_winner(
            run_root=run_root,
            state_path=state_path,
            state=state,
            ref_cfg=ref_cfg,
            run_id=run_id,
            seed=int(args.seed),
            pretest_queries=int(args.sp3_pretest_queries),
            w_ttft=float(args.w_ttft),
            w_rss=float(args.w_rss),
            sampling_interval_ms=int(args.sampling_interval_ms),
            timeseries_stride=int(args.timeseries_stride),
            profile_power=args.profile_power,
            power_sampling_interval_ms=power_sampling_interval_ms,
            ui_update_every=int(args.ui_update_every),
        )
        base_cfg = _with_sp3_profile(base_cfg, winner_profile)

    stage1_rows: list[dict[str, Any]] = [r for r in rows if r.get("stage") == "stage1"]
    if "1" in requested_stages:
        stage1_candidates = candidates
        if int(args.stage1_max_configs) > 0:
            stage1_candidates = candidates[: int(args.stage1_max_configs)]
        for cand in stage1_candidates:
            cfg = _config_from_candidate(base_cfg, cand)
            cfg_id = config_fingerprint(cfg)
            if _find_row(rows, "stage1", cfg_id) is not None:
                continue
            out_dir = run_root / "stage1" / cfg_id
            summary = run_qa_benchmark(
                cfg=cfg,
                dataset="hotpot_qa",
                tier="A",
                output_dir=out_dir,
                run_id=run_id,
                seed=int(args.seed),
                max_queries_override=int(args.stage1_queries),
                retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                profile_timeseries=bool(args.profile_timeseries),
                sampling_interval_ms=int(args.sampling_interval_ms),
                timeseries_stride=int(args.timeseries_stride),
                profile_power=args.profile_power,
                power_sampling_interval_ms=power_sampling_interval_ms,
                ui_update_every=int(args.ui_update_every),
            )
            summary_path = str(out_dir / "summary.json")
            write_summary_markdown(out_dir / "summary.md", summary)
            row = _row_from_summary(
                summary=summary,
                candidate=cand,
                stage="stage1",
                tier="A",
                dataset="hotpot_qa",
                w_ttft=float(args.w_ttft),
                w_rss=float(args.w_rss),
                summary_path=summary_path,
            )
            rows.append(row)
            stage1_rows.append(row)
            state["rows"] = rows
            _save_state(state_path, state)

        # bm25_only references (not in selection)
        if not bool(args.skip_bm25_ref):
            for bm_cfg, meta in _bm25_baselines(base_cfg, [5, 10, 20]):
                cfg_id = config_fingerprint(bm_cfg)
                if _find_row(rows, "stage1_bm25_ref", cfg_id) is not None:
                    continue
                out_dir = run_root / "stage1_bm25_ref" / cfg_id
                summary = run_qa_benchmark(
                    cfg=bm_cfg,
                    dataset="hotpot_qa",
                    tier="A",
                    output_dir=out_dir,
                    run_id=run_id,
                    seed=int(args.seed),
                    max_queries_override=int(args.stage1_queries),
                    retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                    profile_timeseries=bool(args.profile_timeseries),
                    sampling_interval_ms=int(args.sampling_interval_ms),
                    timeseries_stride=int(args.timeseries_stride),
                    profile_power=args.profile_power,
                    power_sampling_interval_ms=power_sampling_interval_ms,
                    ui_update_every=int(args.ui_update_every),
                )
                summary_path = str(out_dir / "summary.json")
                write_summary_markdown(out_dir / "summary.md", summary)
                row = _row_from_summary(
                    summary=summary,
                    candidate=meta,
                    stage="stage1_bm25_ref",
                    tier="A",
                    dataset="hotpot_qa",
                    w_ttft=float(args.w_ttft),
                    w_rss=float(args.w_rss),
                    summary_path=summary_path,
                )
                rows.append(row)
                state["rows"] = rows
                _save_state(state_path, state)

        main_rows = [r for r in rows if r.get("stage") == "stage1"]
        survivors = _pick_survivors(main_rows, keep=6)
        state["stage1_survivors"] = [r["config_id"] for r in survivors]
        _save_state(state_path, state)

    stage2_rows: list[dict[str, Any]] = [r for r in rows if r.get("stage") == "stage2"]
    if "2" in requested_stages:
        survivor_ids = state.get("stage1_survivors", [])
        if int(args.stage2_max_configs) > 0:
            survivor_ids = survivor_ids[: int(args.stage2_max_configs)]
        by_id = {config_fingerprint(_config_from_candidate(base_cfg, c)): c for c in candidates}
        for cfg_id in survivor_ids:
            cand = by_id[cfg_id]
            cfg = _config_from_candidate(base_cfg, cand)
            if _find_row(rows, "stage2", cfg_id) is not None:
                continue
            out_dir = run_root / "stage2" / cfg_id
            summary = run_qa_benchmark(
                cfg=cfg,
                dataset="hotpot_qa",
                tier="B",
                output_dir=out_dir,
                run_id=run_id,
                seed=int(args.seed),
                max_queries_override=int(args.stage2_queries),
                retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                profile_timeseries=bool(args.profile_timeseries),
                sampling_interval_ms=int(args.sampling_interval_ms),
                timeseries_stride=int(args.timeseries_stride),
                profile_power=args.profile_power,
                power_sampling_interval_ms=power_sampling_interval_ms,
                ui_update_every=int(args.ui_update_every),
            )
            summary_path = str(out_dir / "summary.json")
            write_summary_markdown(out_dir / "summary.md", summary)
            row = _row_from_summary(
                summary=summary,
                candidate=cand,
                stage="stage2",
                tier="B",
                dataset="hotpot_qa",
                w_ttft=float(args.w_ttft),
                w_rss=float(args.w_rss),
                summary_path=summary_path,
            )
            rows.append(row)
            stage2_rows.append(row)
            state["rows"] = rows
            _save_state(state_path, state)

        # bm25_only references tier B
        if not bool(args.skip_bm25_ref):
            for bm_cfg, meta in _bm25_baselines(base_cfg, [5, 10, 20]):
                cfg_id = config_fingerprint(bm_cfg)
                if _find_row(rows, "stage2_bm25_ref", cfg_id) is not None:
                    continue
                out_dir = run_root / "stage2_bm25_ref" / cfg_id
                summary = run_qa_benchmark(
                    cfg=bm_cfg,
                    dataset="hotpot_qa",
                    tier="B",
                    output_dir=out_dir,
                    run_id=run_id,
                    seed=int(args.seed),
                    max_queries_override=int(args.stage2_queries),
                    retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                    profile_timeseries=bool(args.profile_timeseries),
                    sampling_interval_ms=int(args.sampling_interval_ms),
                    timeseries_stride=int(args.timeseries_stride),
                    profile_power=args.profile_power,
                    power_sampling_interval_ms=power_sampling_interval_ms,
                    ui_update_every=int(args.ui_update_every),
                )
                summary_path = str(out_dir / "summary.json")
                write_summary_markdown(out_dir / "summary.md", summary)
                row = _row_from_summary(
                    summary=summary,
                    candidate=meta,
                    stage="stage2_bm25_ref",
                    tier="B",
                    dataset="hotpot_qa",
                    w_ttft=float(args.w_ttft),
                    w_rss=float(args.w_rss),
                    summary_path=summary_path,
                )
                rows.append(row)
                state["rows"] = rows
                _save_state(state_path, state)

        main_rows = [r for r in rows if r.get("stage") == "stage2"]
        survivors = _pick_survivors(main_rows, keep=3)
        state["stage2_survivors"] = [r["config_id"] for r in survivors]
        _save_state(state_path, state)

    if "3" in requested_stages:
        stage2_candidates = [r for r in rows if r.get("stage") == "stage2"]
        if not stage2_candidates:
            raise RuntimeError("Stage 2 results are required before stage 3.")
        _mark_pareto(stage2_candidates)
        best_quality = sorted(stage2_candidates, key=lambda x: (x["f1"], x["em"], -x["ttft_ms"]), reverse=True)[0]
        pareto = [r for r in stage2_candidates if r.get("pareto_optimal")]
        best_eff = sorted(pareto, key=lambda x: x["score"], reverse=True)[0]
        if best_eff["config_id"] == best_quality["config_id"]:
            alternatives = [r for r in sorted(stage2_candidates, key=lambda x: x["score"], reverse=True) if r["config_id"] != best_quality["config_id"]]
            if alternatives:
                best_eff = alternatives[0]

        state["best_quality_config_id"] = best_quality["config_id"]
        state["best_efficiency_config_id"] = best_eff["config_id"]
        _save_state(state_path, state)

        by_id = {config_fingerprint(_config_from_candidate(base_cfg, c)): c for c in candidates}
        finalists = [("best_quality", best_quality["config_id"]), ("best_efficiency", best_eff["config_id"])]
        final_cfgs: dict[str, dict[str, Any]] = {}

        for label, cfg_id in finalists:
            cand = by_id[cfg_id]
            cfg = _config_from_candidate(base_cfg, cand)
            final_cfgs[label] = cfg
            out_dir = run_root / "stage3_hotpot_full" / label
            if _find_row(rows, f"stage3_hotpot_full_{label}", cfg_id) is None:
                summary = run_qa_benchmark(
                    cfg=cfg,
                    dataset="hotpot_qa",
                    tier="C",
                    output_dir=out_dir,
                    run_id=run_id,
                    seed=int(args.seed),
                    max_queries_override=7405,
                    retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                    profile_timeseries=bool(args.profile_timeseries),
                    sampling_interval_ms=int(args.sampling_interval_ms),
                    timeseries_stride=int(args.timeseries_stride),
                    profile_power=args.profile_power,
                    power_sampling_interval_ms=power_sampling_interval_ms,
                    ui_update_every=int(args.ui_update_every),
                )
                write_summary_markdown(out_dir / "summary.md", summary)
                row = _row_from_summary(
                    summary=summary,
                    candidate=cand,
                    stage=f"stage3_hotpot_full_{label}",
                    tier="C",
                    dataset="hotpot_qa",
                    w_ttft=float(args.w_ttft),
                    w_rss=float(args.w_rss),
                    summary_path=str(out_dir / "summary.json"),
                )
                rows.append(row)
                state["rows"] = rows
                _save_state(state_path, state)

            # Single-hop NQ fixed 1000
            nq_dir = run_root / "stage3_nq_1000" / label
            if _find_row(rows, f"stage3_nq_{label}", cfg_id) is None:
                nq_summary = run_qa_benchmark(
                    cfg=cfg,
                    dataset="natural_questions",
                    tier="B",
                    output_dir=nq_dir,
                    run_id=run_id,
                    seed=int(args.seed),
                    max_queries_override=1000,
                    retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                    profile_timeseries=bool(args.profile_timeseries),
                    sampling_interval_ms=int(args.sampling_interval_ms),
                    timeseries_stride=int(args.timeseries_stride),
                    profile_power=args.profile_power,
                    power_sampling_interval_ms=power_sampling_interval_ms,
                    ui_update_every=int(args.ui_update_every),
                )
                write_summary_markdown(nq_dir / "summary.md", nq_summary)
                row = _row_from_summary(
                    summary=nq_summary,
                    candidate=cand,
                    stage=f"stage3_nq_{label}",
                    tier="B",
                    dataset="natural_questions",
                    w_ttft=float(args.w_ttft),
                    w_rss=float(args.w_rss),
                    summary_path=str(nq_dir / "summary.json"),
                )
                rows.append(row)
                state["rows"] = rows
                _save_state(state_path, state)

        # BEIR retrieval-only for baseline + finalists
        beir_names = ["scifact", "nfcorpus", "fiqa", "arguana", "quora"]
        beir_cfgs = [("baseline", base_cfg)] + [(label, cfg) for label, cfg in final_cfgs.items()]
        for label, cfg in beir_cfgs:
            cfg_id = config_fingerprint(cfg)
            stage_tag = f"stage3_beir_{label}"
            if _find_row(rows, stage_tag, cfg_id) is not None:
                continue
            beir_dir = run_root / "stage3_beir" / label
            beir_summary = run_beir_retrieval_benchmark(
                cfg=cfg,
                beir_datasets=beir_names,
                tier="B",
                output_dir=beir_dir,
                run_id=run_id,
                seed=int(args.seed),
                max_queries_override=250,
                retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                profile_timeseries=bool(args.profile_timeseries),
                sampling_interval_ms=int(args.sampling_interval_ms),
                timeseries_stride=int(args.timeseries_stride),
                profile_power=args.profile_power,
                power_sampling_interval_ms=power_sampling_interval_ms,
                ui_update_every=int(args.ui_update_every),
            )
            write_summary_markdown(beir_dir / "summary.md", beir_summary)
            row = {
                "stage": stage_tag,
                "tier": "B",
                "dataset": "beir",
                "config_id": cfg_id,
                "retriever_mode": cfg["retrieval"].get("retriever_mode", ""),
                "fusion_method": cfg["retrieval"].get("fusion_method", ""),
                "k": int(cfg["retrieval"].get("top_k_final", 0)),
                "reranker_enabled": bool(cfg["reranker"].get("enabled", False)),
                "rerank_top_in": int(cfg["reranker"].get("top_k_in", 0)),
                "context_packing": bool(cfg["llm"].get("context_packing", False)),
                "prompt_variant": "n/a",
                "max_tokens": int(cfg["llm"].get("max_new_tokens", 0)),
                "f1": 0.0,
                "em": 0.0,
                "ttft_ms": 0.0,
                "rss_peak_mb": float(beir_summary.get("resources", {}).get("rss_peak_bytes", {}).get("mean", 0.0))
                / (1024 * 1024),
                "score": float(beir_summary.get("macro_avg", {}).get("nDCG@10", 0.0)),
                "pareto_optimal": False,
                "summary_path": str(beir_dir / "summary.json"),
            }
            rows.append(row)
            state["rows"] = rows
            _save_state(state_path, state)

        # Save finalist configs
        write_config(run_root / "best_quality_config.yaml", final_cfgs["best_quality"])
        write_config(run_root / "best_efficiency_config.yaml", final_cfgs["best_efficiency"])

        # Optional alpha sweep (cheap) on best hybrid candidate
        if args.run_alpha_sweep and final_cfgs["best_quality"]["retrieval"].get("retriever_mode") == "hybrid":
            alpha_rows = []
            for alpha in [0.3, 0.5, 0.7]:
                cfg = copy.deepcopy(final_cfgs["best_quality"])
                cfg["retrieval"]["fusion_method"] = "WEIGHTED_SUM"
                cfg["retrieval"]["weighted_alpha"] = float(alpha)
                cfg_id = config_fingerprint(cfg)
                alpha_dir = run_root / "alpha_sweep" / f"alpha_{alpha:.1f}_{cfg_id}"
                summary = run_qa_benchmark(
                    cfg=cfg,
                    dataset="hotpot_qa",
                    tier="A",
                    output_dir=alpha_dir,
                    run_id=run_id,
                    seed=int(args.seed),
                    max_queries_override=200,
                    retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                    profile_timeseries=False,
                    sampling_interval_ms=int(args.sampling_interval_ms),
                    timeseries_stride=int(args.timeseries_stride),
                    profile_power=args.profile_power,
                    power_sampling_interval_ms=power_sampling_interval_ms,
                    ui_update_every=int(args.ui_update_every),
                )
                alpha_rows.append(
                    {
                        "alpha": alpha,
                        "EM": float(summary.get("generation", {}).get("EM", 0.0)),
                        "F1": float(summary.get("generation", {}).get("F1", 0.0)),
                        "nDCG@10": float(summary.get("retrieval_metrics", {}).get("nDCG@10", 0.0)),
                    }
                )
            save_json(run_root / "alpha_sweep_results.json", alpha_rows)
            try:
                import matplotlib.pyplot as plt

                alphas = [r["alpha"] for r in alpha_rows]
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(alphas, [r["F1"] for r in alpha_rows], marker="o", label="F1")
                ax.plot(alphas, [r["EM"] for r in alpha_rows], marker="o", label="EM")
                ax.plot(alphas, [r["nDCG@10"] for r in alpha_rows], marker="o", label="nDCG@10")
                ax.set_xlabel("alpha")
                ax.set_ylabel("score")
                ax.set_title("Quality vs weighted_sum alpha")
                ax.legend()
                fig.tight_layout()
                (run_root / "plots").mkdir(parents=True, exist_ok=True)
                fig.savefig(run_root / "plots" / "quality_vs_alpha.png", dpi=150)
                plt.close(fig)
            except Exception:
                pass

    _mark_pareto([r for r in rows if r.get("stage") in {"stage1", "stage2"}])
    state["rows"] = rows
    _save_state(state_path, state)
    _write_leaderboard(run_root, rows)

    save_json(
        run_root / "autotune_summary.json",
        {
            "run_id": run_id,
            "best_quality_config_id": state.get("best_quality_config_id", ""),
            "best_efficiency_config_id": state.get("best_efficiency_config_id", ""),
            "sp3_global_winner": state.get("sp3_global_winner", ""),
            "num_rows": len(rows),
            "stage1_survivors": state.get("stage1_survivors", []),
            "stage2_survivors": state.get("stage2_survivors", []),
        },
    )
    print(f"Done. run_id={run_id}")
    print(f"Output: {run_root}")


if __name__ == "__main__":
    main()
