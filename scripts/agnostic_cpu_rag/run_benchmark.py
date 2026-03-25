#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from agnostic_cpu_rag.adapters.datasets import make_dataset_adapter
from agnostic_cpu_rag.adapters.tasks import make_task_adapter
from agnostic_cpu_rag.config import (
    apply_task_family_profile,
    deep_merge,
    resolve_utility_weights_source,
    resolved_utility_weights,
)
from agnostic_cpu_rag.evaluation import evaluate_query, evaluate_retrieval_run, summarize_query_records
from agnostic_cpu_rag.model_registry import ModelRegistry
from agnostic_cpu_rag.pipeline import AgnosticCPURAGPipeline, write_run_artifacts
from agnostic_cpu_rag.records import RunManifestV2


def load_yaml(path: str) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def atomic_write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def running_status(
    *,
    dataset: str,
    processed: int,
    total: int,
    em_values: list[float],
    f1_values: list[float],
    ttft_values: list[float],
    total_values: list[float],
) -> str:
    parts = [f"{dataset}"]
    if f1_values:
        parts.append(f"F1 {mean(f1_values):.3f}")
    if em_values:
        parts.append(f"EM {mean(em_values):.3f}")
    if ttft_values:
        parts.append(f"TTFT p50 {median(ttft_values):.0f}ms")
    if total_values:
        parts.append(f"T p50 {median(total_values):.0f}ms")
    parts.append(f"{processed}/{total}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agnostic CPU RAG benchmark pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-override", default=None)
    parser.add_argument("--model-registry", default="configs/agnostic_cpu_rag/model_registry.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--task-family", default=None)
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-qids-path", default=None)
    parser.add_argument("--run-id", default="agnostic_smoke")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--beir-name", default=None)
    parser.add_argument("--dataset-streaming", action="store_true")
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--ui-update-every", type=int, default=5)
    parser.add_argument("--pool-role", default="unspecified")
    parser.add_argument("--weights-source", default=None)
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    cfg = dict(base_cfg)
    if args.config_override:
        override_cfg = load_yaml(args.config_override)
        cfg = deep_merge(cfg, override_cfg)
        base_cfg = deep_merge(base_cfg, override_cfg)
    dataset_cfg = dict(cfg.get("dataset", {}))
    dataset_name = str(args.dataset or dataset_cfg.get("name", "hotpot_qa"))
    task_family = str(args.task_family or cfg.get("task", {}).get("family", "auto"))
    max_queries = int(args.num_queries or dataset_cfg.get("num_queries", 50))
    seed = int(args.seed if args.seed is not None else cfg.get("experiment", {}).get("seed", 42))
    include_qids = None
    if args.include_qids_path:
        include_qids = json.loads(Path(args.include_qids_path).read_text(encoding="utf-8"))

    adapter_kwargs: dict[str, Any] = {}
    if args.dataset_streaming or bool(dataset_cfg.get("streaming", False)):
        adapter_kwargs["streaming"] = True
    if dataset_name == "beir":
        adapter_kwargs["beir_name"] = str(args.beir_name or dataset_cfg.get("beir_name", "scifact"))
        adapter_kwargs["data_root"] = str(dataset_cfg.get("data_root", "data/beir"))
    dataset_adapter = make_dataset_adapter(dataset_name, **adapter_kwargs)
    bundle = dataset_adapter.load(
        max_queries=max_queries,
        seed=seed,
        include_qids=include_qids,
        split=str(dataset_cfg.get("split", dataset_adapter.default_split)),
        **adapter_kwargs,
    )
    if task_family == "auto":
        task_family = str(bundle.task_family_hint)
    cfg = apply_task_family_profile(cfg, task_family)
    resolved_weights = resolved_utility_weights(cfg)
    weights_source = str(args.weights_source or resolve_utility_weights_source(base_cfg, task_family))
    task_adapter = make_task_adapter(task_family)
    model_registry = ModelRegistry.load(args.model_registry)

    pipeline = AgnosticCPURAGPipeline(
        cfg=cfg,
        bundle=bundle,
        task_adapter=task_adapter,
        model_registry=model_registry,
        enable_llm=not args.disable_llm,
    )

    run_path = Path(args.output_dir) / args.run_id / dataset_name
    run_path.mkdir(parents=True, exist_ok=True)
    per_query_path = run_path / "per_query.jsonl"
    per_query_path.write_text("", encoding="utf-8")

    records: list[dict[str, Any]] = []
    rankings: dict[str, list[str]] = {}
    sampled_qids = [query.qid for query in bundle.queries]
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
        resolved_utility_weights=dict(resolved_weights),
        artifacts={
            "per_query": str(run_path / "per_query.jsonl"),
            "summary": str(run_path / "summary.json"),
            "summary_partial": str(run_path / "summary.partial.json"),
            "sampled_qids": str(run_path / "sampled_qids.json"),
        },
        notes={
            "disable_llm": bool(args.disable_llm),
            "dataset_streaming": bool(adapter_kwargs.get("streaming", False)),
            "config_override_path": str(args.config_override) if args.config_override else None,
        },
    )
    atomic_write_json(run_path / "run_manifest.json", manifest.to_dict())
    atomic_write_json(run_path / "sampled_qids.json", sampled_qids)

    ui_every = max(1, int(args.ui_update_every))
    em_values: list[float] = []
    f1_values: list[float] = []
    ttft_values: list[float] = []
    total_values: list[float] = []

    with per_query_path.open("a", encoding="utf-8") as per_query_handle, Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        task_id = progress.add_task(
            running_status(
                dataset=dataset_name,
                processed=0,
                total=len(bundle.queries),
                em_values=em_values,
                f1_values=f1_values,
                ttft_values=ttft_values,
                total_values=total_values,
            ),
            total=len(bundle.queries),
        )
        for idx, query_record in enumerate(bundle.queries, start=1):
            gold = bundle.gold_references.get(query_record.qid)
            result = pipeline.run_query(query_record, gold=gold)
            record = result.to_record()
            if gold is not None:
                record["metrics"] = evaluate_query(
                    task_adapter=task_adapter,
                    gold=gold,
                    prediction=result.prediction,
                    selected_doc_ids=result.selected_doc_ids,
                )
            metrics = record.get("metrics", {})
            if metrics.get("em") is not None:
                em_values.append(float(metrics["em"]))
            if metrics.get("f1") is not None:
                f1_values.append(float(metrics["f1"]))
            latency = record.get("latency_ms", {})
            if latency.get("ttft_ms") is not None:
                ttft_values.append(float(latency["ttft_ms"]))
            if latency.get("t_total_ms") is not None:
                total_values.append(float(latency["t_total_ms"]))

            rankings[query_record.qid] = list(result.retrieved_doc_ids)
            records.append(record)
            per_query_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            per_query_handle.flush()

            should_refresh = (idx % ui_every == 0) or (idx == len(bundle.queries))
            progress.update(
                task_id,
                advance=1,
                description=running_status(
                    dataset=dataset_name,
                    processed=idx,
                    total=len(bundle.queries),
                    em_values=em_values,
                    f1_values=f1_values,
                    ttft_values=ttft_values,
                    total_values=total_values,
                ),
            )
            if should_refresh:
                partial_summary = summarize_query_records(records)
                partial_summary["processed_queries"] = idx
                partial_summary["total_queries"] = len(bundle.queries)
                partial_summary["complete"] = bool(idx == len(bundle.queries))
                atomic_write_json(run_path / "summary.partial.json", partial_summary)

    summary = summarize_query_records(records)
    summary["retrieval"] = evaluate_retrieval_run(
        rankings=rankings,
        bundle=bundle,
        ks=list(cfg.get("evaluation", {}).get("retrieval_ks", [1, 2, 5, 10, 20])),
    )["aggregate"]

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
        resolved_utility_weights=dict(resolved_weights),
        artifacts={
            "per_query": str(run_path / "per_query.jsonl"),
            "summary": str(run_path / "summary.json"),
            "summary_partial": str(run_path / "summary.partial.json"),
            "sampled_qids": str(run_path / "sampled_qids.json"),
        },
        notes={
            "disable_llm": bool(args.disable_llm),
            "config_override_path": str(args.config_override) if args.config_override else None,
        },
    )
    write_run_artifacts(
        output_dir=str(run_path),
        run_manifest=manifest.to_dict(),
        records=records,
        summary=summary,
        sampled_qids=sampled_qids,
    )
    print(json.dumps({"run_dir": str(run_path), "num_queries": len(bundle.queries)}, indent=2))


if __name__ == "__main__":
    main()
