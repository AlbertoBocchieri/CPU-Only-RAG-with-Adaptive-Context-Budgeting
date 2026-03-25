#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_json(path: str) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_yaml(path: str) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def llm_registry_path(model_registry: dict[str, Any], llm_name: str) -> str:
    return str(
        (((model_registry.get("models") or {}).get("llm") or {}).get(llm_name) or {}).get("gguf_path", "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce a parity matrix between the new agnostic pipeline and the legacy rag_cpu pipeline.")
    parser.add_argument("--new-config", default="configs/agnostic_cpu_rag/fixed_alpha_nohop2_multihop_stable.yaml")
    parser.add_argument("--new-model-registry", default="configs/agnostic_cpu_rag/model_registry.yaml")
    parser.add_argument("--legacy-effective-config", default="results/incremental_stop_full_7405_power_p4/cfg_c9ea9dbfd0ed/effective_config.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    new_cfg = load_yaml(args.new_config)
    registry = load_yaml(args.new_model_registry)
    legacy_cfg = load_json(args.legacy_effective_config)

    new_llm_name = str((new_cfg.get("llm") or {}).get("model_name", "qwen25_3b_q4km"))
    new_llm_path = llm_registry_path(registry, new_llm_name)
    legacy_llm_path = str((legacy_cfg.get("llm") or {}).get("gguf_path", ""))

    new_components = {
        "dense_model": str((new_cfg.get("retrieval") or {}).get("dense_model_name", "")),
        "dense_backend_target": str((((registry.get("models") or {}).get("dense") or {}).get(str((new_cfg.get("retrieval") or {}).get("dense_model_name", ""))) or {}).get("model_name", "")),
        "fusion_method": "weighted_sum_conf_fixed",
        "fusion_alpha": float((new_cfg.get("retrieval") or {}).get("fixed_alpha", 0.5)),
        "hop2_enabled": bool((((new_cfg.get("retrieval") or {}).get("hop2") or {}).get("enabled", False))),
        "top_k_final": int((new_cfg.get("retrieval") or {}).get("top_k_final", 0)),
        "controller_strategy": str((new_cfg.get("context_controller") or {}).get("stop_mode", "")),
        "llm_path": new_llm_path,
        "llm_prompt_family": "answer_template_for_task",
        "llm_temperature": float((new_cfg.get("llm") or {}).get("temperature", 0.0)),
        "llm_max_new_tokens": int((new_cfg.get("llm") or {}).get("max_new_tokens", 0)),
        "llm_threads_decode": int((new_cfg.get("llm") or {}).get("threads_decode", 0)),
        "llm_threads_batch": int((new_cfg.get("llm") or {}).get("threads_batch", 0)),
        "llm_batch_size": int((new_cfg.get("llm") or {}).get("batch_size", 0)),
    }
    legacy_components = {
        "dense_model": str((legacy_cfg.get("retrieval") or {}).get("dense_model", "")),
        "dense_backend_target": str((legacy_cfg.get("retrieval") or {}).get("dense_model", "")),
        "fusion_method": str((legacy_cfg.get("retrieval") or {}).get("fusion_method", "")),
        "fusion_alpha": float((legacy_cfg.get("retrieval") or {}).get("weighted_alpha", 0.0)),
        "hop2_enabled": bool((legacy_cfg.get("retrieval") or {}).get("multi_hop_enabled", False)),
        "top_k_final": int((legacy_cfg.get("retrieval") or {}).get("top_k_final", 0)),
        "controller_strategy": str((legacy_cfg.get("context_budgeting") or {}).get("strategy", "")),
        "llm_path": legacy_llm_path,
        "llm_prompt_family": str((legacy_cfg.get("llm") or {}).get("prompt_mode", "")),
        "llm_temperature": float((legacy_cfg.get("llm") or {}).get("temperature", 0.0)),
        "llm_max_new_tokens": int((legacy_cfg.get("llm") or {}).get("max_new_tokens", 0)),
        "llm_threads_decode": int(((legacy_cfg.get("llm_runtime") or {}).get("threads_decode", 0))),
        "llm_threads_batch": int(((legacy_cfg.get("llm_runtime") or {}).get("threads_batch", 0))),
        "llm_batch_size": int(((legacy_cfg.get("llm_runtime") or {}).get("batch_size", 512) or 512)),
    }

    rows = [
        {
            "component": "dense_retrieval",
            "new": {"model_alias": new_components["dense_model"], "target": new_components["dense_backend_target"]},
            "legacy": {"model": legacy_components["dense_model"]},
            "status": "different",
            "quality_affecting": True,
            "reason": "Different dense encoders imply different candidate sets and score distributions.",
        },
        {
            "component": "fusion",
            "new": {"method": new_components["fusion_method"], "alpha": new_components["fusion_alpha"]},
            "legacy": {"method": legacy_components["fusion_method"], "alpha": legacy_components["fusion_alpha"]},
            "status": "different",
            "quality_affecting": True,
            "reason": "Alpha 0.5 vs 0.75 changes dense/BM25 weighting.",
        },
        {
            "component": "hop2_query_expansion",
            "new": {"enabled": new_components["hop2_enabled"]},
            "legacy": {"enabled": legacy_components["hop2_enabled"]},
            "status": "different",
            "quality_affecting": True,
            "reason": "Legacy published pipeline keeps multi-hop expansion enabled; new stable disables it.",
        },
        {
            "component": "context_controller",
            "new": {"strategy": new_components["controller_strategy"]},
            "legacy": {"strategy": legacy_components["controller_strategy"]},
            "status": "different",
            "quality_affecting": True,
            "reason": "Controller logic is the main treatment and not comparable as identical.",
        },
        {
            "component": "llm_weights",
            "new": {"gguf_path": new_components["llm_path"]},
            "legacy": {"gguf_path": legacy_components["llm_path"]},
            "status": ("identical" if new_components["llm_path"] == legacy_components["llm_path"] else "different"),
            "quality_affecting": new_components["llm_path"] != legacy_components["llm_path"],
            "reason": "Both pipelines use the same Qwen2.5 3B GGUF file." if new_components["llm_path"] == legacy_components["llm_path"] else "Different model weights.",
        },
        {
            "component": "prompting",
            "new": {
                "template_family": new_components["llm_prompt_family"],
                "temperature": new_components["llm_temperature"],
                "max_new_tokens": new_components["llm_max_new_tokens"],
            },
            "legacy": {
                "template_family": legacy_components["llm_prompt_family"],
                "temperature": legacy_components["llm_temperature"],
                "max_new_tokens": legacy_components["llm_max_new_tokens"],
            },
            "status": "different",
            "quality_affecting": True,
            "reason": "Prompt template wording and decoding settings differ.",
        },
        {
            "component": "runtime_threads",
            "new": {
                "threads_decode": new_components["llm_threads_decode"],
                "threads_batch": new_components["llm_threads_batch"],
                "batch_size": new_components["llm_batch_size"],
            },
            "legacy": {
                "threads_decode": legacy_components["llm_threads_decode"],
                "threads_batch": legacy_components["llm_threads_batch"],
                "batch_size": legacy_components["llm_batch_size"],
            },
            "status": ("identical" if (
                new_components["llm_threads_decode"] == legacy_components["llm_threads_decode"]
                and new_components["llm_threads_batch"] == legacy_components["llm_threads_batch"]
                and new_components["llm_batch_size"] == legacy_components["llm_batch_size"]
            ) else "different"),
            "quality_affecting": False,
            "reason": "Runtime knobs primarily affect latency; they are not treated as the controller intervention.",
        },
        {
            "component": "evaluation_schema",
            "new": {"schema": "agnostic_cpu_rag summary/per_query"},
            "legacy": {"schema": "rag_cpu summary/per_query"},
            "status": "different",
            "quality_affecting": False,
            "reason": "Reporting paths differ, so legacy runs must be normalized before comparison.",
        },
    ]

    pipeline_level_required = any(row["quality_affecting"] and row["status"] != "identical" for row in rows if row["component"] != "context_controller")
    out = {
        "new_config_path": str(args.new_config),
        "new_model_registry_path": str(args.new_model_registry),
        "legacy_effective_config_path": str(args.legacy_effective_config),
        "rows": rows,
        "conclusion": {
            "pipeline_level_required": bool(pipeline_level_required),
            "controller_level_allowed": bool(not pipeline_level_required),
            "recommended_framing": (
                "pipeline-level comparison"
                if pipeline_level_required
                else "controller-level comparison permissible"
            ),
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
