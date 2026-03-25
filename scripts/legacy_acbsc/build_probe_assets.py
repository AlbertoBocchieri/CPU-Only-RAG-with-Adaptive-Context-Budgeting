#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_cpu.config import deep_update, load_config, write_config

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs/base.yaml"
OUT_DIR = ROOT / "configs/legacy_acbsc_probe"
QID_DIR = ROOT / "qids"

COMMON_RETRIEVAL = {
    "dense_model": "intfloat/e5-base-v2",
    "dense_prefix_query": "query: ",
    "dense_prefix_passage": "passage: ",
    "retriever_mode": "hybrid",
    "fusion_method": "WEIGHTED_SUM",
    "weighted_alpha": 0.75,
    "normalize_scores": True,
    "agreement_bonus_enabled": False,
    "top_k_dense": 150,
    "top_k_bm25": 250,
    "top_k_final": 20,
    "multi_hop_enabled": True,
    "multi_hop_mode": "hybrid",
    "multi_hop_top_seed_hits": 2,
    "multi_hop_max_entities": 4,
    "multi_hop_merge_rrf_k": 60,
    "multi_hop_gate_enabled": False,
}

COMMON_RUNTIME = {
    "sp3_enabled": False,
    "sp3_profile": "BASELINE",
    "threads_decode": 4,
    "threads_batch": 4,
    "batch_size": None,
    "ubatch_size": None,
}

LEGACY_INCREMENTAL_SC = {
    "enabled": True,
    "strategy": "incremental_sc",
    "keep_full_count": 2,
    "seed_min_unique_docs": 3,
    "seed_token_fraction": 0.35,
    "snippet_from_rank": 4,
    "prefer_full_until_rank": 3,
    "snippet_window_tokens": 80,
    "full_relevance_floor": 0.60,
    "utility_relevance_weight": 0.55,
    "utility_question_overlap_weight": 0.20,
    "utility_novelty_weight": 0.15,
    "utility_new_doc_weight": 0.10,
    "prefill_target_ms": 18000,
    "cap_bootstrap_tokens": 1024,
    "cap_min_tokens": 768,
    "cap_max_tokens": 1536,
    "warmup_queries": 8,
    "ewma_alpha": 0.2,
    "use_rerank_scores_if_available": True,
    "max_chunks_hard_cap": 20,
}

AGNOSTIC_ACB_SC = {
    "enabled": True,
    "strategy": "agnostic_acb_sc",
    "stop_mode": "coverage_locked_patience_v3",
    "seed_min_items": 3,
    "required_distinct_docs": 2,
    "snippet_words": 120,
    "min_snippet_words": 40,
    "utility_relevance_weight": 0.55,
    "utility_question_overlap_weight": 0.25,
    "utility_novelty_weight": 0.10,
    "utility_new_doc_weight": 0.10,
    "patience": 2,
    "single_evidence_extra_probe_candidates": 0,
    "multi_document_spare_probe_candidates": 1,
    "multi_document_exact_probe_candidates": 2,
    "marginal_snippet_enabled": True,
    "marginal_snippet_ratio": 0.60,
    "marginal_snippet_min_words": 40,
    "marginal_snippet_max_words": 72,
    "comparison_guard_enabled": False,
    "comparison_guard_extra_probe_enabled": False,
    "prefill_target_ms": 18000,
    "cap_bootstrap_tokens": 1024,
    "cap_min_tokens": 768,
    "cap_max_tokens": 1536,
    "warmup_queries": 8,
    "ewma_alpha": 0.2,
    "use_rerank_scores_if_available": True,
    "max_chunks_hard_cap": 20,
}

DATASET_SETTINGS: dict[str, dict[str, Any]] = {
    "hotpot_qa": {
        "llm": {"n_threads": 4},
        "probe_qids_source": ROOT / "results/task_family_weight_search/pools/hotpot_qa/holdout_1000_qids.json",
        "probe_qids_out": QID_DIR / "hotpot_legacy_acbsc_probe100.json",
        "representative_qids_source": ROOT / "results/task_family_weight_search/pools/hotpot_qa/representative_300_qids.json",
        "representative_qids_out": QID_DIR / "hotpot_legacy_acbsc_rep300.json",
    },
    "two_wiki_multihop": {
        "llm": {"n_threads": 4, "temperature": 0.0, "max_new_tokens": 32},
        "probe_qids_source": ROOT / "results/task_family_weight_search/pools/two_wiki_multihop/holdout_1000_qids.json",
        "probe_qids_out": QID_DIR / "twowiki_legacy_acbsc_probe100.json",
        "representative_qids_source": ROOT / "results/task_family_weight_search/pools/two_wiki_multihop/representative_300_qids.json",
        "representative_qids_out": QID_DIR / "twowiki_legacy_acbsc_rep300.json",
    },
}


def build_qids(source_path: Path, output_path: Path, count: int) -> list[str]:
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    qids = [str(row) for row in rows[:count]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(qids, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return qids



def build_config(dataset_name: str, context_budgeting: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config(BASE_CONFIG)
    cfg = deep_update(
        cfg,
        {
            "datasets": {
                "qa": {"name": dataset_name},
                dataset_name: {"split": "validation"},
            },
            "retrieval": dict(COMMON_RETRIEVAL),
            "reranker": {"enabled": False},
            "llm": dict(DATASET_SETTINGS[dataset_name]["llm"]),
            "llm_runtime": dict(COMMON_RUNTIME),
            "context_budgeting": dict(context_budgeting),
        },
    )
    return cfg



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {"configs": {}, "probe_qids": {}, "representative_qids": {}}
    for dataset_name, settings in DATASET_SETTINGS.items():
        prefix = "hotpot" if dataset_name == "hotpot_qa" else "twowiki"
        config_specs = {
            f"{prefix}_noacb_p4.yaml": {"enabled": False},
            f"{prefix}_incremental_sc_p4.yaml": LEGACY_INCREMENTAL_SC,
            f"{prefix}_agnostic_acb_sc_p4.yaml": AGNOSTIC_ACB_SC,
        }
        for filename, cb_cfg in config_specs.items():
            out_path = OUT_DIR / filename
            write_config(out_path, build_config(dataset_name, cb_cfg))
            outputs["configs"][filename] = str(out_path.relative_to(ROOT))
        qids = build_qids(settings["probe_qids_source"], settings["probe_qids_out"], count=100)
        outputs["probe_qids"][dataset_name] = {
            "source": str(settings["probe_qids_source"].relative_to(ROOT)),
            "output": str(settings["probe_qids_out"].relative_to(ROOT)),
            "count": len(qids),
            "first_qid": (qids[0] if qids else ""),
            "last_qid": (qids[-1] if qids else ""),
        }
        rep_qids = build_qids(settings["representative_qids_source"], settings["representative_qids_out"], count=300)
        outputs["representative_qids"][dataset_name] = {
            "source": str(settings["representative_qids_source"].relative_to(ROOT)),
            "output": str(settings["representative_qids_out"].relative_to(ROOT)),
            "count": len(rep_qids),
            "first_qid": (rep_qids[0] if rep_qids else ""),
            "last_qid": (rep_qids[-1] if rep_qids else ""),
        }
    print(json.dumps(outputs, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
