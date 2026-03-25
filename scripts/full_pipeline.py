#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rag_cpu.autotune import run_autotune
from rag_cpu.benchmark import benchmark_beir_retrieval, benchmark_qa
from rag_cpu.config import load_config, write_config
from rag_cpu.stats import bootstrap_ci, paired_permutation_test
from rag_cpu.utils import now_ts, save_json, set_seed


def _apply_quick_mode(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["datasets"]["qa"]["train_size"] = 120
    out["datasets"]["qa"]["val_size"] = 80
    out["datasets"]["qa"]["test_size"] = 100
    out["datasets"]["qa"]["max_corpus_docs"] = 900

    out["datasets"]["beir"]["max_queries_per_dataset"] = 80

    out["benchmark"]["generation_max_samples"] = 20
    out["benchmark"]["bootstrap_samples"] = 600
    out["benchmark"]["permutation_trials"] = 600

    out["autotune"]["trials_retrieval"] = 6
    out["autotune"]["trials_generation"] = 3
    return out


def _metric_list_from_retrieval(result: dict[str, Any], metric_name: str) -> list[float]:
    per_q = result.get("retrieval_per_query", {})
    values = [vals.get(metric_name, 0.0) for vals in per_q.values()]
    return values


def _qa_stat_block(
    baseline: dict[str, Any],
    tuned: dict[str, Any],
    bootstrap_samples: int,
    permutation_trials: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    b_em = baseline.get("generation", {}).get("em_values", [])
    t_em = tuned.get("generation", {}).get("em_values", [])
    b_f1 = baseline.get("generation", {}).get("f1_values", [])
    t_f1 = tuned.get("generation", {}).get("f1_values", [])

    out["EM"] = {
        "baseline_ci": bootstrap_ci(b_em, n_samples=bootstrap_samples, seed=seed),
        "tuned_ci": bootstrap_ci(t_em, n_samples=bootstrap_samples, seed=seed),
        "permutation": paired_permutation_test(
            t_em,
            b_em,
            n_trials=permutation_trials,
            seed=seed,
        ),
    }
    out["F1"] = {
        "baseline_ci": bootstrap_ci(b_f1, n_samples=bootstrap_samples, seed=seed),
        "tuned_ci": bootstrap_ci(t_f1, n_samples=bootstrap_samples, seed=seed),
        "permutation": paired_permutation_test(
            t_f1,
            b_f1,
            n_trials=permutation_trials,
            seed=seed,
        ),
    }

    for metric in ("Recall@5", "nDCG@10", "MRR@10"):
        b = _metric_list_from_retrieval(baseline, metric)
        t = _metric_list_from_retrieval(tuned, metric)
        out[metric] = {
            "baseline_ci": bootstrap_ci(b, n_samples=bootstrap_samples, seed=seed),
            "tuned_ci": bootstrap_ci(t, n_samples=bootstrap_samples, seed=seed),
            "permutation": paired_permutation_test(
                t,
                b,
                n_trials=permutation_trials,
                seed=seed,
            ),
        }

    return out


def _format_metrics_line(metrics: dict[str, float], keys: list[str]) -> str:
    parts = []
    for k in keys:
        parts.append(f"{k}: {metrics.get(k, 0.0):.4f}")
    return " | ".join(parts)


def _write_markdown_report(
    out_path: Path,
    cfg: dict[str, Any],
    baseline_qa: dict[str, Any],
    tuned_qa: dict[str, Any],
    stats_block: dict[str, Any],
    autotune: dict[str, Any],
    baseline_beir: dict[str, Any] | None,
    tuned_beir: dict[str, Any] | None,
    run_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# CPU-only RAG Report (Qwen 3B)")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Model GGUF: `{cfg['llm']['gguf_path']}`")
    lines.append("- Hardware target: CPU-only")
    lines.append("")
    lines.append("## Baseline vs Tuned on QA Test")
    lines.append("")
    b_ret = baseline_qa["retrieval_metrics"]
    t_ret = tuned_qa["retrieval_metrics"]
    lines.append(f"- Baseline retrieval: {_format_metrics_line(b_ret, ['Recall@5', 'nDCG@10', 'MRR@10'])}")
    lines.append(f"- Tuned retrieval: {_format_metrics_line(t_ret, ['Recall@5', 'nDCG@10', 'MRR@10'])}")

    b_gen = baseline_qa.get("generation", {})
    t_gen = tuned_qa.get("generation", {})
    lines.append(
        f"- Baseline generation: EM {b_gen.get('EM', 0.0):.4f} | F1 {b_gen.get('F1', 0.0):.4f} | latency mean {b_gen.get('latency_s', {}).get('mean', 0.0):.3f}s"
    )
    lines.append(
        f"- Tuned generation: EM {t_gen.get('EM', 0.0):.4f} | F1 {t_gen.get('F1', 0.0):.4f} | latency mean {t_gen.get('latency_s', {}).get('mean', 0.0):.3f}s"
    )

    lines.append("")
    lines.append("## Statistical Significance")
    lines.append("")
    for metric in ["EM", "F1", "Recall@5", "nDCG@10", "MRR@10"]:
        m = stats_block[metric]
        p = m["permutation"]["p_value"]
        diff = m["permutation"]["diff_mean"]
        lines.append(
            f"- {metric}: diff(tuned-baseline)={diff:.4f}, p-value={p:.5f}, "
            f"baseline CI [{m['baseline_ci']['ci_low']:.4f}, {m['baseline_ci']['ci_high']:.4f}], "
            f"tuned CI [{m['tuned_ci']['ci_low']:.4f}, {m['tuned_ci']['ci_high']:.4f}]"
        )

    lines.append("")
    lines.append("## Autotuning Summary")
    lines.append("")
    lines.append(f"- Retrieval trials: {len(autotune.get('retrieval_trials', []))}")
    lines.append(f"- Generation trials: {len(autotune.get('generation_trials', []))}")
    lines.append(f"- Best validation objective: {autotune.get('best', {}).get('score', 0.0):.4f}")

    lines.append("")
    lines.append("## BEIR Retrieval")
    lines.append("")
    if baseline_beir and tuned_beir:
        lines.append(
            f"- Baseline macro: {_format_metrics_line(baseline_beir.get('macro_avg', {}), ['Recall@5', 'nDCG@10', 'MRR@10'])}"
        )
        lines.append(
            f"- Tuned macro: {_format_metrics_line(tuned_beir.get('macro_avg', {}), ['Recall@5', 'nDCG@10', 'MRR@10'])}"
        )
    else:
        lines.append("- BEIR step skipped")

    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- Saved configs and raw outputs in this run directory.")
    lines.append("- For long final runs use the commands in `RUN_COMMANDS.md`.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full CPU-only RAG pipeline")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-beir", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode == "quick":
        cfg = _apply_quick_mode(cfg)

    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    run_dir = Path(cfg["experiment"]["output_dir"]) / f"run_{now_ts()}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir / "effective_config.yaml", cfg)

    print("[1/6] Baseline QA test benchmark (with generation)...")
    baseline_qa = benchmark_qa(
        cfg,
        split="test",
        enable_llm=True,
        max_generation_samples=int(cfg["benchmark"]["generation_max_samples"]),
    )
    save_json(run_dir / "baseline_qa.json", baseline_qa)

    print("[2/6] Autotuning on validation split...")
    autotune = run_autotune(cfg, output_dir=run_dir / "autotune")

    print("[3/6] Evaluate tuned config on QA test...")
    best_cfg = autotune["best"]["config"]
    write_config(run_dir / "best_config.yaml", best_cfg)
    tuned_qa = benchmark_qa(
        best_cfg,
        split="test",
        enable_llm=True,
        max_generation_samples=int(best_cfg["benchmark"]["generation_max_samples"]),
    )
    save_json(run_dir / "tuned_qa.json", tuned_qa)

    print("[4/6] Statistical tests...")
    stats_block = _qa_stat_block(
        baseline=baseline_qa,
        tuned=tuned_qa,
        bootstrap_samples=int(cfg["benchmark"]["bootstrap_samples"]),
        permutation_trials=int(cfg["benchmark"]["permutation_trials"]),
        seed=seed,
    )
    save_json(run_dir / "qa_stats.json", stats_block)

    baseline_beir = None
    tuned_beir = None
    if not args.skip_beir:
        print("[5/6] Baseline BEIR retrieval...")
        baseline_beir = benchmark_beir_retrieval(cfg)
        save_json(run_dir / "baseline_beir.json", baseline_beir)

        print("[6/6] Tuned BEIR retrieval...")
        tuned_beir = benchmark_beir_retrieval(best_cfg)
        save_json(run_dir / "tuned_beir.json", tuned_beir)
    else:
        print("[5/6] BEIR skipped by flag")
        print("[6/6] BEIR skipped by flag")

    summary = {
        "baseline_qa": {
            "retrieval": baseline_qa["retrieval_metrics"],
            "generation": {
                "EM": baseline_qa["generation"]["EM"],
                "F1": baseline_qa["generation"]["F1"],
            },
        },
        "tuned_qa": {
            "retrieval": tuned_qa["retrieval_metrics"],
            "generation": {
                "EM": tuned_qa["generation"]["EM"],
                "F1": tuned_qa["generation"]["F1"],
            },
        },
        "qa_stats": stats_block,
        "beir": {
            "baseline_macro": baseline_beir.get("macro_avg", {}) if baseline_beir else {},
            "tuned_macro": tuned_beir.get("macro_avg", {}) if tuned_beir else {},
        },
    }
    save_json(run_dir / "summary.json", summary)

    _write_markdown_report(
        out_path=run_dir / "REPORT.md",
        cfg=cfg,
        baseline_qa=baseline_qa,
        tuned_qa=tuned_qa,
        stats_block=stats_block,
        autotune=autotune,
        baseline_beir=baseline_beir,
        tuned_beir=tuned_beir,
        run_dir=run_dir,
    )

    print("Done.")
    print(f"Run dir: {run_dir}")
    print(
        "Key metrics | baseline F1: "
        f"{baseline_qa['generation']['F1']:.4f}, tuned F1: {tuned_qa['generation']['F1']:.4f}"
    )


if __name__ == "__main__":
    main()
