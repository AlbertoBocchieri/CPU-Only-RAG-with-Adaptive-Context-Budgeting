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

from rag_cpu.benchmark_suite import (
    config_fingerprint,
    run_beir_retrieval_benchmark,
    run_qa_benchmark,
    write_summary_markdown,
)
from rag_cpu.config import deep_update, load_config
from rag_cpu.utils import now_ts, save_json


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _plot_outputs(run_root: Path, summaries: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = run_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    qa_entries: list[tuple[str, list[dict[str, Any]]]] = []
    for s in summaries:
        per_query = s.get("artifacts", {}).get("per_query_jsonl", "")
        rows = _load_jsonl(Path(per_query))
        if rows:
            label = f"{s.get('dataset','')}:{s.get('config_id','')[:6]}"
            qa_entries.append((label, rows))

    # EM/F1 vs TTFT
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, rows in qa_entries:
        ttft = [float(r.get("latency_ms", {}).get("ttft_ms", 0.0)) for r in rows]
        em = [float(r.get("answer_metrics_per_query", {}).get("em", 0.0)) for r in rows]
        f1 = [float(r.get("answer_metrics_per_query", {}).get("f1", 0.0)) for r in rows]
        if not any(ttft):
            continue
        ax.scatter(ttft, f1, s=10, alpha=0.45, label=f"F1 {label}")
        ax.scatter(ttft, em, s=10, alpha=0.25, label=f"EM {label}")
    ax.set_xlabel("TTFT (ms)")
    ax.set_ylabel("Score")
    ax.set_title("EM/F1 vs TTFT")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "em_f1_vs_ttft_scatter.png", dpi=150)
    plt.close(fig)

    # EM/F1 vs total latency
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, rows in qa_entries:
        t_total = [float(r.get("latency_ms", {}).get("t_total_ms", 0.0)) for r in rows]
        em = [float(r.get("answer_metrics_per_query", {}).get("em", 0.0)) for r in rows]
        f1 = [float(r.get("answer_metrics_per_query", {}).get("f1", 0.0)) for r in rows]
        if not any(t_total):
            continue
        ax.scatter(t_total, f1, s=10, alpha=0.45, label=f"F1 {label}")
        ax.scatter(t_total, em, s=10, alpha=0.25, label=f"EM {label}")
    ax.set_xlabel("Total latency (ms)")
    ax.set_ylabel("Score")
    ax.set_title("EM/F1 vs Total Latency")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "em_f1_vs_t_total_scatter.png", dpi=150)
    plt.close(fig)

    # EM/F1 vs context tokens
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, rows in qa_entries:
        ctx_tok = [float(r.get("tokens", {}).get("context_tokens", 0.0)) for r in rows]
        em = [float(r.get("answer_metrics_per_query", {}).get("em", 0.0)) for r in rows]
        f1 = [float(r.get("answer_metrics_per_query", {}).get("f1", 0.0)) for r in rows]
        if not any(ctx_tok):
            continue
        ax.scatter(ctx_tok, f1, s=10, alpha=0.45, label=f"F1 {label}")
        ax.scatter(ctx_tok, em, s=10, alpha=0.25, label=f"EM {label}")
    ax.set_xlabel("Context tokens")
    ax.set_ylabel("Score")
    ax.set_title("EM/F1 vs Context Tokens")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "em_f1_vs_context_tokens.png", dpi=150)
    plt.close(fig)

    # Recall@k curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in summaries:
        metrics = s.get("retrieval_metrics", s.get("macro_avg", {}))
        points: list[tuple[int, float]] = []
        for key, value in metrics.items():
            if key.startswith("Recall@"):
                try:
                    k = int(key.split("@")[1])
                    points.append((k, float(value)))
                except Exception:
                    continue
        if not points:
            continue
        points.sort(key=lambda x: x[0])
        label = f"{s.get('dataset','')}:{s.get('config_id','')[:6]}"
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=label)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k Curves")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "recall_at_k_curves.png", dpi=150)
    plt.close(fig)

    # Latency breakdown stacked bars
    labels: list[str] = []
    retrieval_vals: list[float] = []
    rerank_vals: list[float] = []
    prefill_vals: list[float] = []
    decode_vals: list[float] = []
    for s in summaries:
        lat = s.get("latency_ms", {})
        if not lat:
            continue
        labels.append(f"{s.get('dataset','')}:{s.get('config_id','')[:6]}")
        retrieval_vals.append(float(lat.get("t_retrieval_total_ms", {}).get("mean", 0.0)))
        rerank_vals.append(float(lat.get("t_rerank_total_ms", {}).get("mean", 0.0)))
        prefill_vals.append(float(lat.get("t_prefill_ms", {}).get("mean", 0.0)))
        decode_vals.append(float(lat.get("t_decode_total_ms", {}).get("mean", 0.0)))
    if labels:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(len(labels)))
        ax.bar(x, retrieval_vals, label="retrieval")
        ax.bar(x, rerank_vals, bottom=retrieval_vals, label="rerank")
        base2 = [a + b for a, b in zip(retrieval_vals, rerank_vals, strict=False)]
        ax.bar(x, prefill_vals, bottom=base2, label="prefill")
        base3 = [a + b for a, b in zip(base2, prefill_vals, strict=False)]
        ax.bar(x, decode_vals, bottom=base3, label="decode")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("ms (mean)")
        ax.set_title("Latency Breakdown")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "latency_breakdown_stacked_bars.png", dpi=150)
        plt.close(fig)

    # Resource bars
    labels = []
    rss_vals = []
    cpu_vals = []
    for s in summaries:
        res = s.get("resources", {})
        if not res:
            continue
        labels.append(f"{s.get('dataset','')}:{s.get('config_id','')[:6]}")
        rss_vals.append(float(res.get("rss_peak_bytes", {}).get("mean", 0.0)) / (1024 * 1024))
        cpu_vals.append(float(res.get("cpu_mean_pct", {}).get("mean", res.get("cpu_mean_pct", {}).get("p50", 0.0))))
    if labels:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        x = list(range(len(labels)))
        ax1.bar(x, rss_vals, color="#5B8FF9", alpha=0.8, label="RSS peak (MB)")
        ax1.set_ylabel("RSS peak (MB)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax2 = ax1.twinx()
        ax2.plot(x, cpu_vals, color="#F4664A", marker="o", label="CPU mean (%)")
        ax2.set_ylabel("CPU mean (%)")
        ax1.set_title("Resource Usage per Config")
        fig.tight_layout()
        fig.savefig(plots_dir / "resource_rss_cpu_per_config.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark suite with instrumentation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-override", default="")
    parser.add_argument("--config-2", default="")
    parser.add_argument("--config-3", default="")
    parser.add_argument("--dataset", choices=["hotpot_qa", "natural_questions", "two_wiki_multihop", "beir"], required=True)
    parser.add_argument("--tier", choices=["A", "B", "C"], default="A")
    parser.add_argument("--num-queries", type=int, default=0)
    parser.add_argument("--beir-datasets", default="scifact,nfcorpus,fiqa,arguana,quora")
    parser.add_argument("--beir-max-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--query-ids-path", default="")
    parser.add_argument("--profile-timeseries", action="store_true")
    parser.add_argument("--sampling-interval-ms", type=int, default=200)
    parser.add_argument("--timeseries-stride", type=int, default=5)
    parser.add_argument("--profile-power", dest="profile_power", action="store_true")
    parser.add_argument("--no-profile-power", dest="profile_power", action="store_false")
    parser.set_defaults(profile_power=None)
    parser.add_argument("--power-sampling-interval-ms", type=int, default=0)
    parser.add_argument("--ui-update-every", type=int, default=5)
    args = parser.parse_args()

    config_paths = [args.config]
    if args.config_2:
        config_paths.append(args.config_2)
    if args.config_3:
        config_paths.append(args.config_3)

    run_id = args.run_id or f"bench_{now_ts()}_{args.dataset}_{args.tier.lower()}"
    run_root = Path(args.output_dir) / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    runs_meta: list[dict[str, Any]] = []

    override_cfg: dict[str, Any] = {}
    if args.config_override:
        override_cfg = load_config(args.config_override)

    for cfg_path in config_paths:
        cfg = load_config(cfg_path)
        if override_cfg:
            cfg = deep_update(cfg, override_cfg)
        config_id = config_fingerprint(cfg)
        config_dir = run_root / f"cfg_{config_id}"
        config_dir.mkdir(parents=True, exist_ok=True)
        save_json(config_dir / "effective_config.json", cfg)

        if args.dataset in {"hotpot_qa", "natural_questions", "two_wiki_multihop"}:
            summary = run_qa_benchmark(
                cfg=cfg,
                dataset=args.dataset,
                tier=args.tier,
                output_dir=config_dir / args.dataset,
                run_id=run_id,
                seed=int(args.seed),
                max_queries_override=(int(args.num_queries) if args.num_queries > 0 else None),
                retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                profile_timeseries=bool(args.profile_timeseries),
                sampling_interval_ms=int(args.sampling_interval_ms),
                timeseries_stride=int(args.timeseries_stride),
                profile_power=args.profile_power,
                power_sampling_interval_ms=(
                    int(args.power_sampling_interval_ms) if int(args.power_sampling_interval_ms) > 0 else None
                ),
                ui_update_every=int(args.ui_update_every),
                query_ids_path=(args.query_ids_path or None),
            )
            summary["config_path"] = cfg_path
            save_json(config_dir / args.dataset / "summary.json", summary)
            write_summary_markdown(config_dir / args.dataset / "summary.md", summary)
            summaries.append(summary)
            runs_meta.append(
                {
                    "config_id": config_id,
                    "config_path": cfg_path,
                    "dataset": args.dataset,
                    "summary_path": str(config_dir / args.dataset / "summary.json"),
                }
            )
        else:
            beir_names = [x.strip() for x in args.beir_datasets.split(",") if x.strip()]
            summary = run_beir_retrieval_benchmark(
                cfg=cfg,
                beir_datasets=beir_names,
                tier=args.tier,
                output_dir=config_dir / "beir",
                run_id=run_id,
                seed=int(args.seed),
                max_queries_override=(int(args.beir_max_queries) if args.beir_max_queries > 0 else None),
                retrieval_ks=[1, 2, 5, 10, 20, 50, 100],
                profile_timeseries=bool(args.profile_timeseries),
                sampling_interval_ms=int(args.sampling_interval_ms),
                timeseries_stride=int(args.timeseries_stride),
                profile_power=args.profile_power,
                power_sampling_interval_ms=(
                    int(args.power_sampling_interval_ms) if int(args.power_sampling_interval_ms) > 0 else None
                ),
                ui_update_every=int(args.ui_update_every),
            )
            summary["config_path"] = cfg_path
            save_json(config_dir / "beir" / "summary.json", summary)
            write_summary_markdown(config_dir / "beir" / "summary.md", summary)
            summaries.append(summary)
            runs_meta.append(
                {
                    "config_id": config_id,
                    "config_path": cfg_path,
                    "dataset": "beir",
                    "summary_path": str(config_dir / "beir" / "summary.json"),
                }
            )

    merged = {"run_id": run_id, "dataset": args.dataset, "tier": args.tier, "runs": runs_meta, "summaries": summaries}
    save_json(run_root / "summary.json", merged)

    lines = ["# Benchmark Suite Summary", "", f"- run_id: `{run_id}`", f"- dataset: `{args.dataset}`", f"- tier: `{args.tier}`", ""]
    for s in summaries:
        lines.append(f"## {s.get('dataset','')} / {s.get('config_id','')}")
        lines.append("")
        lines.append(f"- retriever_mode: `{s.get('retriever_mode','')}`")
        lines.append(f"- fusion_method: `{s.get('fusion_method','')}`")
        lines.append(f"- rrf_k: `{s.get('rrf_k','')}`")
        lines.append(f"- agreement_bonus_enabled: `{s.get('agreement_bonus_enabled', False)}`")
        lines.append(f"- agreement_bonus: `{s.get('agreement_bonus', 0.0)}`")
        lines.append(f"- weighted_alpha: `{s.get('weighted_alpha', None)}`")
        sp3 = s.get("sp3", {})
        lines.append(f"- sp3_profile: `{sp3.get('profile', '')}`")
        lines.append(f"- threads_decode/batch: `{sp3.get('threads_decode', '')}/{sp3.get('threads_batch', '')}`")
        if "generation" in s:
            lines.append(f"- EM: {s['generation'].get('EM', 0.0):.4f}")
            lines.append(f"- F1: {s['generation'].get('F1', 0.0):.4f}")
        if "context_budgeting" in s:
            cb = s["context_budgeting"]
            lines.append(f"- ACB enabled: {bool(cb.get('enabled', False))}")
            lines.append(f"- ACB strategy: `{cb.get('strategy', 'v1')}`")
            lines.append(
                f"- ACB low/medium/high branch: {cb.get('pct_low_branch', 0.0):.3f}/"
                f"{cb.get('pct_medium_branch', 0.0):.3f}/{cb.get('pct_high_branch', 0.0):.3f}"
            )
            lines.append(f"- ACB fallback_to_high: {cb.get('pct_fallback_to_high', 0.0):.3f}")
        if "retrieval_metrics" in s:
            lines.append(f"- Recall@5: {s['retrieval_metrics'].get('Recall@5', 0.0):.4f}")
            lines.append(f"- nDCG@10: {s['retrieval_metrics'].get('nDCG@10', 0.0):.4f}")
        supp = s.get("supporting_docs", {})
        if isinstance(supp, dict):
            pair = supp.get("pair_recall", {})
            if isinstance(pair, dict):
                pr5 = pair.get("pair_recall@5", None)
                pr10 = pair.get("pair_recall@10", None)
                if pr5 is not None:
                    lines.append(f"- pair_recall@5: {float(pr5):.4f}")
                if pr10 is not None:
                    lines.append(f"- pair_recall@10: {float(pr10):.4f}")
        if "macro_avg" in s:
            lines.append(f"- BEIR macro Recall@5: {s['macro_avg'].get('Recall@5', 0.0):.4f}")
            lines.append(f"- BEIR macro nDCG@10: {s['macro_avg'].get('nDCG@10', 0.0):.4f}")
        lines.append("")
    (run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    _plot_outputs(run_root, summaries)
    print(f"Done. run_id={run_id}")
    print(f"Output: {run_root}")


if __name__ == "__main__":
    main()
