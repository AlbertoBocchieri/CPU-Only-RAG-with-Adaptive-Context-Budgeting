#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from agnostic_cpu_rag.adapters.datasets import make_dataset_adapter
from agnostic_cpu_rag.adapters.tasks import make_task_adapter
from agnostic_cpu_rag.config import apply_task_family_profile, deep_merge, resolve_utility_weights_source, resolved_utility_weights
from agnostic_cpu_rag.model_registry import ModelRegistry
from agnostic_cpu_rag.pipeline import AgnosticCPURAGPipeline
from agnostic_cpu_rag.records import RunManifestV2
from agnostic_cpu_rag.weight_search import serialize_candidate, serialize_gold


def load_yaml(path: str) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export retrieval/controller cache rows for offline utility-weight evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--model-registry", default="configs/agnostic_cpu_rag/model_registry.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-family", default=None)
    parser.add_argument("--include-qids-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="results/task_family_weight_search/caches")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pool-role", default="tuning")
    parser.add_argument("--weights-source", default=None)
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    cfg = dict(base_cfg)
    if args.config_override:
        override_cfg = load_yaml(args.config_override)
        cfg = deep_merge(cfg, override_cfg)
        base_cfg = deep_merge(base_cfg, override_cfg)

    include_qids = json.loads(Path(args.include_qids_path).read_text(encoding="utf-8"))
    dataset_cfg = dict(cfg.get("dataset", {}))
    seed = int(args.seed if args.seed is not None else cfg.get("experiment", {}).get("seed", 42))
    dataset_name = str(args.dataset)
    adapter_kwargs: dict[str, Any] = {}
    dataset_adapter = make_dataset_adapter(dataset_name, **adapter_kwargs)
    bundle = dataset_adapter.load(
        max_queries=len(include_qids),
        seed=seed,
        include_qids=include_qids,
        split=str(dataset_cfg.get("split", dataset_adapter.default_split)),
        **adapter_kwargs,
    )
    task_family = str(args.task_family or bundle.task_family_hint)
    cfg = apply_task_family_profile(cfg, task_family)
    weights_source = str(args.weights_source or resolve_utility_weights_source(base_cfg, task_family))
    model_registry = ModelRegistry.load(args.model_registry)
    pipeline = AgnosticCPURAGPipeline(
        cfg=cfg,
        bundle=bundle,
        task_adapter=make_task_adapter(task_family),
        model_registry=model_registry,
        enable_llm=False,
    )

    out_dir = Path(args.output_dir) / args.run_id / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_rows: list[dict[str, Any]] = []
    cache_path = out_dir / "controller_cache.jsonl"
    cache_path.write_text("", encoding="utf-8")

    with cache_path.open("a", encoding="utf-8") as handle, Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        task_id = progress.add_task(f"cache {dataset_name}", total=len(bundle.queries))
        for query_record in bundle.queries:
            retrieval = pipeline.retriever.search(query=query_record.query, calibrator=pipeline.calibrator)
            gold = bundle.gold_references[query_record.qid]
            runtime_snapshot = pipeline.calibrator.snapshot()
            row = {
                "qid": query_record.qid,
                "query": query_record.query,
                "task_family": task_family,
                "coverage_goal": pipeline.task_adapter.coverage_goal.value,
                "required_distinct_docs": pipeline.task_adapter.required_distinct_docs(),
                "budget_cap_tokens": int(runtime_snapshot.get("budget_cap_tokens", 0)),
                "budget_cap_source": str(runtime_snapshot.get("budget_cap_source", "bootstrap")),
                "gold": serialize_gold(gold),
                "candidates": [serialize_candidate(candidate) for candidate in retrieval.candidates],
                "retrieval_trace": retrieval.trace,
            }
            cache_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            progress.advance(task_id)

    manifest = RunManifestV2(
        run_id=str(args.run_id),
        dataset=str(dataset_name),
        task_family=str(task_family),
        pool_role=str(args.pool_role),
        split=str(dataset_cfg.get("split", dataset_adapter.default_split)),
        seed=int(seed),
        num_queries=len(bundle.queries),
        config_path=str(args.config),
        model_registry_path=str(args.model_registry),
        weights_source=str(weights_source),
        resolved_utility_weights=resolved_utility_weights(cfg),
        artifacts={
            "controller_cache": str(cache_path),
            "sampled_qids": str(out_dir / "sampled_qids.json"),
        },
        notes={
            "disable_llm": True,
            "config_override_path": str(args.config_override) if args.config_override else None,
            "source_qids_path": str(args.include_qids_path),
        },
    )
    atomic_write_json(out_dir / "sampled_qids.json", [query.qid for query in bundle.queries])
    atomic_write_json(out_dir / "cache_manifest.json", manifest.to_dict())
    print(json.dumps({"cache_dir": str(out_dir), "num_queries": len(cache_rows)}, indent=2))


if __name__ == "__main__":
    main()
