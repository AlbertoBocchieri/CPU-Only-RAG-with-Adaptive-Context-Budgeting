#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rag_cpu.benchmark import benchmark_qa
from rag_cpu.config import load_config
from rag_cpu.stats import bootstrap_ci, paired_permutation_test
from rag_cpu.utils import now_ts, save_json, set_seed


def _out_dir_from_config(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    if parent.name.startswith("run_"):
        return parent
    return Path("results") / f"run_{now_ts()}_prompt_ablation"


def _summary_block(result: dict[str, Any]) -> dict[str, float]:
    gen = result.get("generation", {})
    ret = result.get("retrieval_metrics", {})
    return {
        "EM": float(gen.get("EM", 0.0)),
        "F1": float(gen.get("F1", 0.0)),
        "latency_mean_s": float(gen.get("latency_s", {}).get("mean", 0.0)),
        "retrieval_recall5": float(ret.get("Recall@5", 0.0)),
        "retrieval_ndcg10": float(ret.get("nDCG@10", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B prompt ablation: rag_strict vs direct")
    parser.add_argument("--config", required=True, help="YAML config to evaluate")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--max-generation-samples",
        type=int,
        default=None,
        help="Override generation sample count",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSON artifacts (default: run dir from config path)",
    )
    parser.add_argument(
        "--direct-template",
        default=None,
        help="Optional direct prompt template override",
    )
    parser.add_argument("--perm-trials", type=int, default=10000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    set_seed(int(cfg["experiment"]["seed"]))

    out_dir = Path(args.output_dir) if args.output_dir else _out_dir_from_config(cfg_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    rag_cfg = copy.deepcopy(cfg)
    rag_cfg["llm"]["prompt_mode"] = "rag_strict"

    direct_cfg = copy.deepcopy(cfg)
    direct_cfg["llm"]["prompt_mode"] = "direct"
    if args.direct_template is not None:
        direct_cfg["llm"]["direct_prompt_template"] = args.direct_template

    print("Running rag_strict prompt eval...")
    rag = benchmark_qa(
        rag_cfg,
        split=args.split,
        enable_llm=True,
        max_generation_samples=args.max_generation_samples,
    )
    rag_path = out_dir / f"prompt_rag_strict_{args.split}.json"
    save_json(rag_path, rag)

    print("Running direct prompt eval...")
    direct = benchmark_qa(
        direct_cfg,
        split=args.split,
        enable_llm=True,
        max_generation_samples=args.max_generation_samples,
    )
    direct_path = out_dir / f"prompt_direct_{args.split}.json"
    save_json(direct_path, direct)

    rag_summary = _summary_block(rag)
    direct_summary = _summary_block(direct)

    em_rag = [float(v) for v in rag.get("generation", {}).get("em_values", [])]
    em_direct = [float(v) for v in direct.get("generation", {}).get("em_values", [])]
    f1_rag = [float(v) for v in rag.get("generation", {}).get("f1_values", [])]
    f1_direct = [float(v) for v in direct.get("generation", {}).get("f1_values", [])]

    summary: dict[str, Any] = {
        "rag_strict": rag_summary,
        "direct": direct_summary,
        "delta_direct_minus_rag": {
            "EM": direct_summary["EM"] - rag_summary["EM"],
            "F1": direct_summary["F1"] - rag_summary["F1"],
            "latency_mean_s": direct_summary["latency_mean_s"] - rag_summary["latency_mean_s"],
        },
        "significance": {
            "EM": {
                "permutation": paired_permutation_test(
                    em_direct,
                    em_rag,
                    n_trials=int(args.perm_trials),
                    seed=int(cfg["experiment"]["seed"]),
                ),
                "rag_ci": bootstrap_ci(
                    em_rag,
                    n_samples=int(args.bootstrap_samples),
                    seed=int(cfg["experiment"]["seed"]),
                ),
                "direct_ci": bootstrap_ci(
                    em_direct,
                    n_samples=int(args.bootstrap_samples),
                    seed=int(cfg["experiment"]["seed"]),
                ),
            },
            "F1": {
                "permutation": paired_permutation_test(
                    f1_direct,
                    f1_rag,
                    n_trials=int(args.perm_trials),
                    seed=int(cfg["experiment"]["seed"]),
                ),
                "rag_ci": bootstrap_ci(
                    f1_rag,
                    n_samples=int(args.bootstrap_samples),
                    seed=int(cfg["experiment"]["seed"]),
                ),
                "direct_ci": bootstrap_ci(
                    f1_direct,
                    n_samples=int(args.bootstrap_samples),
                    seed=int(cfg["experiment"]["seed"]),
                ),
            },
        },
        "artifacts": {
            "rag_json": str(rag_path),
            "direct_json": str(direct_path),
        },
    }

    summary_path = out_dir / "prompt_ablation_summary.json"
    save_json(summary_path, summary)

    print("Done prompt ablation")
    print(summary)


if __name__ == "__main__":
    main()
