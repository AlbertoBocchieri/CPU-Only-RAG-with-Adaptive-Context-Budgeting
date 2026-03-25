from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

from .benchmark import benchmark_qa
from .config import write_config
from .utils import save_json


def _sample(rng: random.Random, values: list[Any]) -> Any:
    return values[rng.randrange(len(values))]


def _apply_retrieval_trial(cfg: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["chunking"]["chunk_size_words"] = int(trial["chunk_size_words"])
    out["chunking"]["chunk_overlap_words"] = int(trial["chunk_overlap_words"])

    out["retrieval"]["top_k_dense"] = int(trial["top_k_dense"])
    out["retrieval"]["top_k_bm25"] = int(trial["top_k_bm25"])
    out["retrieval"]["top_k_final"] = int(trial["top_k_final"])
    out["retrieval"]["hybrid_alpha"] = float(trial["hybrid_alpha"])

    out["reranker"]["enabled"] = bool(trial["reranker_enabled"])
    out["reranker"]["top_k_in"] = int(trial["reranker_top_k_in"])
    out["reranker"]["top_k_out"] = int(min(out["retrieval"]["top_k_final"], out["reranker"]["top_k_in"]))

    return out


def _apply_generation_trial(cfg: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["llm"]["temperature"] = float(trial["llm_temperature"])
    out["llm"]["top_p"] = float(trial["llm_top_p"])
    out["llm"]["repeat_penalty"] = float(trial["llm_repeat_penalty"])
    return out


def _retrieval_objective(metrics: dict[str, float], latency_mean_s: float) -> float:
    return (
        0.45 * metrics.get("Recall@5", 0.0)
        + 0.35 * metrics.get("nDCG@10", 0.0)
        + 0.20 * metrics.get("MRR@10", 0.0)
        - 0.01 * latency_mean_s
    )


def _generation_objective(generation: dict[str, Any], retrieval_metrics: dict[str, float]) -> float:
    return (
        0.55 * generation.get("F1", 0.0)
        + 0.20 * generation.get("EM", 0.0)
        + 0.20 * retrieval_metrics.get("Recall@5", 0.0)
        - 0.05 * generation.get("latency_s", {}).get("mean", 0.0)
    )


def run_autotune(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    tuning_cfg = cfg["autotune"]
    space = tuning_cfg["search"]
    rng = random.Random(int(tuning_cfg["random_seed"]))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    retrieval_trials = int(tuning_cfg["trials_retrieval"])
    generation_trials = int(tuning_cfg["trials_generation"])

    retrieval_rows: list[dict[str, Any]] = []

    for t in range(retrieval_trials):
        trial = {
            "chunk_size_words": _sample(rng, space["chunk_size_words"]),
            "chunk_overlap_words": _sample(rng, space["chunk_overlap_words"]),
            "top_k_dense": _sample(rng, space["top_k_dense"]),
            "top_k_bm25": _sample(rng, space["top_k_bm25"]),
            "top_k_final": _sample(rng, space["top_k_final"]),
            "hybrid_alpha": _sample(rng, space["hybrid_alpha"]),
            "reranker_enabled": _sample(rng, space["reranker_enabled"]),
            "reranker_top_k_in": _sample(rng, space["reranker_top_k_in"]),
        }
        trial_cfg = _apply_retrieval_trial(cfg, trial)

        qa_ret = benchmark_qa(trial_cfg, split="val", enable_llm=False)
        score = _retrieval_objective(
            qa_ret["retrieval_metrics"], qa_ret["retrieval_latency_s"]["mean"]
        )

        retrieval_rows.append(
            {
                "trial_id": t,
                "params": trial,
                "score": score,
                "retrieval_metrics": qa_ret["retrieval_metrics"],
                "latency": qa_ret["retrieval_latency_s"],
            }
        )

    retrieval_rows.sort(key=lambda x: x["score"], reverse=True)

    top_retrieval_cfgs: list[dict[str, Any]] = []
    for row in retrieval_rows[: min(6, len(retrieval_rows))]:
        top_retrieval_cfgs.append(_apply_retrieval_trial(cfg, row["params"]))

    generation_rows: list[dict[str, Any]] = []

    for t in range(generation_trials):
        base_cfg = copy.deepcopy(top_retrieval_cfgs[t % len(top_retrieval_cfgs)])
        gen_params = {
            "llm_temperature": _sample(rng, space["llm_temperature"]),
            "llm_top_p": _sample(rng, space["llm_top_p"]),
            "llm_repeat_penalty": _sample(rng, space["llm_repeat_penalty"]),
        }

        trial_cfg = _apply_generation_trial(base_cfg, gen_params)

        qa_gen = benchmark_qa(
            trial_cfg,
            split="val",
            enable_llm=True,
            max_generation_samples=min(60, int(cfg["benchmark"]["generation_max_samples"])),
        )
        gen_obj = _generation_objective(qa_gen["generation"], qa_gen["retrieval_metrics"])

        generation_rows.append(
            {
                "trial_id": t,
                "params": gen_params,
                "base_retrieval_cfg": {
                    "chunking": trial_cfg["chunking"],
                    "retrieval": trial_cfg["retrieval"],
                    "reranker": trial_cfg["reranker"],
                },
                "score": gen_obj,
                "retrieval_metrics": qa_gen["retrieval_metrics"],
                "generation": {
                    k: v
                    for k, v in qa_gen["generation"].items()
                    if k in {"EM", "F1", "num_samples", "latency_s", "prompt_tokens_mean", "completion_tokens_mean"}
                },
            }
        )

    generation_rows.sort(key=lambda x: x["score"], reverse=True)
    best_row = generation_rows[0]

    best_cfg = copy.deepcopy(cfg)
    best_cfg["chunking"] = copy.deepcopy(best_row["base_retrieval_cfg"]["chunking"])
    best_cfg["retrieval"] = copy.deepcopy(best_row["base_retrieval_cfg"]["retrieval"])
    best_cfg["reranker"] = copy.deepcopy(best_row["base_retrieval_cfg"]["reranker"])
    best_cfg = _apply_generation_trial(best_cfg, best_row["params"])

    out = {
        "retrieval_trials": retrieval_rows,
        "generation_trials": generation_rows,
        "best": {
            "score": best_row["score"],
            "config": best_cfg,
            "validation_metrics": {
                "retrieval": best_row["retrieval_metrics"],
                "generation": best_row["generation"],
            },
        },
    }

    save_json(output_path / "autotune_results.json", out)
    write_config(output_path / "best_config.yaml", best_cfg)

    return out
