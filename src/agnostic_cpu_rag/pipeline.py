from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from rag_cpu.generator import LlamaCppGenerator

from .adapters.base import TaskAdapter
from .context_controller import ContextController
from .generation.prompting import answer_sanity, answer_template_for_task
from .model_registry import ModelRegistry
from .records import CoverageGoal, DatasetBundle, GoldReference, QueryRecord
from .retrieval import AgnosticRetriever
from .runtime.calibration import LatencyCalibrator


@dataclass(slots=True)
class PipelineQueryResult:
    qid: str
    query: str
    prediction: str | None
    retrieved_doc_ids: list[str]
    selected_doc_ids: list[str]
    retrieval_trace: dict[str, Any]
    context_controller: dict[str, Any]
    latency_ms: dict[str, Any]
    runtime: dict[str, Any]
    answer_sanity: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "query": self.query,
            "prediction": self.prediction,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "selected_doc_ids": self.selected_doc_ids,
            "retrieval_trace": self.retrieval_trace,
            "context_controller": self.context_controller,
            "latency_ms": self.latency_ms,
            "runtime": self.runtime,
            "answer_sanity": self.answer_sanity,
        }


class AgnosticCPURAGPipeline:
    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        bundle: DatasetBundle,
        task_adapter: TaskAdapter,
        model_registry: ModelRegistry,
        enable_llm: bool | None = None,
    ):
        self.cfg = cfg
        self.bundle = bundle
        self.task_adapter = task_adapter
        self.model_registry = model_registry
        self.process = psutil.Process(os.getpid())

        self.retriever = AgnosticRetriever(
            documents=bundle.documents,
            config=dict(cfg.get("retrieval", {})),
            model_registry=model_registry,
        )
        self.context_controller = ContextController(cfg.get("context_controller", {}))
        runtime_cfg = dict(cfg.get("runtime", {}))
        self.calibrator = LatencyCalibrator(
            prefill_target_ms=float(runtime_cfg.get("prefill_target_ms", 18000.0)),
            cap_min_tokens=int(runtime_cfg.get("cap_min_tokens", 512)),
            cap_max_tokens=int(runtime_cfg.get("cap_max_tokens", 1536)),
            bootstrap_cap_tokens=int(runtime_cfg.get("bootstrap_cap_tokens", 1024)),
            fixed_cap_tokens=(
                int(runtime_cfg["fixed_cap_tokens"])
                if runtime_cfg.get("fixed_cap_tokens", None) is not None
                else None
            ),
            warmup_queries=int(runtime_cfg.get("warmup_queries", 8)),
            ewma_alpha=float(runtime_cfg.get("ewma_alpha", 0.2)),
        )

        llm_cfg = dict(cfg.get("llm", {}))
        self.enable_llm = bool(llm_cfg.get("enabled", True)) if enable_llm is None else bool(enable_llm)
        self.generator: LlamaCppGenerator | None = None
        if self.enable_llm and self.task_adapter.supports_generation:
            llm_name = str(llm_cfg.get("model_name", "qwen25_3b_q4km"))
            llm_spec = model_registry.get("llm", llm_name)
            self.generator = LlamaCppGenerator(
                gguf_path=str(llm_spec.params["gguf_path"]),
                n_ctx=int(llm_cfg.get("n_ctx", llm_spec.params.get("context_length", 4096))),
                n_threads=int(llm_cfg.get("threads_decode", 4)),
                n_batch=int(llm_cfg.get("batch_size", 512)),
                n_threads_batch=int(llm_cfg.get("threads_batch", 4)),
                n_ubatch=(
                    int(llm_cfg["ubatch_size"])
                    if llm_cfg.get("ubatch_size", None) is not None
                    else None
                ),
                prefix_cache_enabled=bool(llm_cfg.get("prefix_cache_enabled", False)),
                prefix_cache_backend=str(llm_cfg.get("prefix_cache_backend", "ram")),
                prefix_cache_capacity_mb=int(llm_cfg.get("prefix_cache_capacity_mb", 256)),
                prefix_cache_dir=str(llm_cfg.get("prefix_cache_dir", "cache/llama_prompt_cache")),
            )

    def run_query(self, query_record: QueryRecord, gold: GoldReference | None = None) -> PipelineQueryResult:
        t0 = time.perf_counter()
        rss_before = float(self.process.memory_info().rss / (1024 * 1024))
        retrieval = self.retriever.search(query=query_record.query, calibrator=self.calibrator)
        runtime_before_gen = self.calibrator.snapshot()

        if self.task_adapter.coverage_goal == CoverageGoal.RETRIEVAL_ONLY or not self.task_adapter.supports_generation:
            selected_doc_ids = [cand.doc_id for cand in retrieval.candidates]
            context_trace = {
                "enabled": False,
                "coverage_goal": str(self.task_adapter.coverage_goal.value),
                "context_doc_ids_used": selected_doc_ids,
                "context_tokens_used": 0,
                "budget_cap_tokens": runtime_before_gen["budget_cap_tokens"],
            }
            prediction = None
            answer_check = {"empty": 0, "too_long": 0, "malformed": 0, "token_count": 0, "looks_valid": False}
            latency_ms = {
                "t_retrieval_total_ms": float(retrieval.trace.get("t_retrieval_total_ms", 0.0)),
                "t_rerank_total_ms": float(retrieval.trace.get("rerank", {}).get("t_rerank_total_ms", 0.0)),
                "t_total_ms": float((time.perf_counter() - t0) * 1000.0),
            }
        elif self.generator is None:
            context = self.context_controller.select(
                query=query_record.query,
                candidates=retrieval.candidates,
                coverage_goal=self.task_adapter.coverage_goal,
                required_distinct_docs=self.task_adapter.required_distinct_docs(),
                budget_cap_tokens=int(runtime_before_gen["budget_cap_tokens"]),
            )
            selected_doc_ids = list(context.trace.get("context_doc_ids_used", []))
            prediction = None
            answer_check = {"empty": 0, "too_long": 0, "malformed": 0, "token_count": 0, "looks_valid": False}
            latency_ms = {
                "t_retrieval_total_ms": float(retrieval.trace.get("t_retrieval_total_ms", 0.0)),
                "t_rerank_total_ms": float(retrieval.trace.get("rerank", {}).get("t_rerank_total_ms", 0.0)),
                "t_total_ms": float((time.perf_counter() - t0) * 1000.0),
            }
            context_trace = dict(context.trace)
        else:
            context = self.context_controller.select(
                query=query_record.query,
                candidates=retrieval.candidates,
                coverage_goal=self.task_adapter.coverage_goal,
                required_distinct_docs=self.task_adapter.required_distinct_docs(),
                budget_cap_tokens=int(runtime_before_gen["budget_cap_tokens"]),
            )
            selected_doc_ids = list(context.trace.get("context_doc_ids_used", []))
            gen = self.generator.generate(
                question=query_record.query,
                contexts=context.contexts,
                temperature=float(self.cfg.get("llm", {}).get("temperature", 0.0)),
                top_p=float(self.cfg.get("llm", {}).get("top_p", 0.9)),
                max_new_tokens=int(self.cfg.get("llm", {}).get("max_new_tokens", 64)),
                repeat_penalty=float(self.cfg.get("llm", {}).get("repeat_penalty", 1.05)),
                prompt_mode="direct",
                direct_template=answer_template_for_task(self.task_adapter.family),
                answer_postprocess_mode=str(self.cfg.get("llm", {}).get("answer_postprocess_mode", "basic")),
                enable_stream_timing=bool(self.cfg.get("llm", {}).get("stream_timing", True)),
            )
            prediction = str(gen.answer)
            answer_check = answer_sanity(prediction)
            latency_ms = {
                "t_retrieval_total_ms": float(retrieval.trace.get("t_retrieval_total_ms", 0.0)),
                "t_rerank_total_ms": float(retrieval.trace.get("rerank", {}).get("t_rerank_total_ms", 0.0)),
                "t_llm_total_ms": float(gen.t_llm_total_ms),
                "t_prefill_ms": float(gen.t_prefill_ms),
                "t_decode_total_ms": float(gen.t_decode_total_ms),
                "ttft_ms": float(gen.ttft_ms),
                "t_total_ms": float((time.perf_counter() - t0) * 1000.0),
            }
            self.calibrator.update_prefill(
                context_tokens=int(context.trace.get("context_tokens_used", 0)),
                prefill_ms=float(gen.t_prefill_ms),
            )
            self.calibrator.update_decode(
                output_tokens=int(gen.completion_tokens),
                decode_ms=float(gen.t_decode_total_ms),
            )
            context_trace = dict(context.trace)

        rss_after = float(self.process.memory_info().rss / (1024 * 1024))
        self.calibrator.update_rss(max(rss_before, rss_after))
        runtime_snapshot = self.calibrator.snapshot()
        runtime_snapshot["rss_mb_before"] = float(rss_before)
        runtime_snapshot["rss_mb_after"] = float(rss_after)
        return PipelineQueryResult(
            qid=query_record.qid,
            query=query_record.query,
            prediction=prediction,
            retrieved_doc_ids=[cand.doc_id for cand in retrieval.candidates],
            selected_doc_ids=selected_doc_ids,
            retrieval_trace=retrieval.trace,
            context_controller=context_trace,
            latency_ms=latency_ms,
            runtime=runtime_snapshot,
            answer_sanity=answer_check,
        )


def write_run_artifacts(
    *,
    output_dir: str,
    run_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    sampled_qids: list[str],
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "sampled_qids.json").write_text(json.dumps(sampled_qids, indent=2), encoding="utf-8")
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
