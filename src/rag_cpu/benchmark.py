from __future__ import annotations

import time
from statistics import mean
from typing import Any

from tqdm import tqdm

from .chunking import chunk_documents
from .data import (
    build_qrels_from_qa_examples,
    load_beir_dataset,
    load_squad_qa,
    map_doc_qrels_to_chunk_qrels,
)
from .metrics import evaluate_retrieval, qa_scores, summarize_list
from .pipeline import RAGPipeline
from .retrievers import chunks_to_items, docs_to_items, unique_doc_ids


def _run_retrieval_loop(
    pipeline: RAGPipeline,
    cfg: dict[str, Any],
    queries: dict[str, str],
    dedupe_to_doc_ids: bool,
) -> tuple[dict[str, list[str]], list[float]]:
    runs: dict[str, list[str]] = {}
    latencies: list[float] = []

    for qid, question in tqdm(queries.items(), desc="Retrieval", leave=False):
        t0 = time.perf_counter()
        hits = pipeline.retrieve(question, cfg)
        latencies.append(time.perf_counter() - t0)

        if dedupe_to_doc_ids:
            runs[qid] = unique_doc_ids(hits)
        else:
            runs[qid] = [h.item_id for h in hits]

    return runs, latencies


def benchmark_qa(
    cfg: dict[str, Any],
    split: str,
    enable_llm: bool,
    max_generation_samples: int | None = None,
) -> dict[str, Any]:
    qa_cfg = cfg["datasets"]["qa"]
    benchmark_cfg = cfg["benchmark"]

    splits, docs = load_squad_qa(
        train_size=int(qa_cfg["train_size"]),
        val_size=int(qa_cfg["val_size"]),
        test_size=int(qa_cfg["test_size"]),
        max_corpus_docs=int(qa_cfg["max_corpus_docs"]),
        seed=int(cfg["experiment"]["seed"]),
    )

    chunks, doc_to_chunks = chunk_documents(
        docs,
        chunk_size_words=int(cfg["chunking"]["chunk_size_words"]),
        chunk_overlap_words=int(cfg["chunking"]["chunk_overlap_words"]),
        min_chunk_words=int(cfg["chunking"]["min_chunk_words"]),
    )

    examples = splits[split]
    doc_qrels = build_qrels_from_qa_examples(examples)
    chunk_qrels = map_doc_qrels_to_chunk_qrels(doc_qrels, doc_to_chunks)

    queries = {ex.qid: ex.question for ex in examples}
    items = chunks_to_items(chunks)
    pipeline = RAGPipeline(cfg=cfg, items=items, enable_llm=enable_llm)

    runs, retrieval_lat = _run_retrieval_loop(
        pipeline=pipeline,
        cfg=cfg,
        queries=queries,
        dedupe_to_doc_ids=False,
    )
    ret_metrics, ret_per_query = evaluate_retrieval(
        runs,
        chunk_qrels,
        ks=[int(k) for k in benchmark_cfg["retrieval_ks"]],
    )

    result: dict[str, Any] = {
        "task": "qa",
        "split": split,
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "num_queries": len(queries),
        "retrieval_metrics": ret_metrics,
        "retrieval_per_query": ret_per_query,
        "retrieval_latency_s": summarize_list(retrieval_lat),
    }

    if not enable_llm:
        return result

    max_samples = max_generation_samples or int(benchmark_cfg["generation_max_samples"])
    eval_examples = examples[:max_samples]

    em_values: list[float] = []
    f1_values: list[float] = []
    gen_lat: list[float] = []
    prompt_toks: list[int] = []
    completion_toks: list[int] = []
    rows: list[dict[str, Any]] = []

    for ex in tqdm(eval_examples, desc="Generation", leave=False):
        t0 = time.perf_counter()
        out = pipeline.answer(ex.question, cfg)
        gen_lat.append(time.perf_counter() - t0)

        pred = out.answer or ""
        s = qa_scores(pred, ex.answers)
        em_values.append(s["em"])
        f1_values.append(s["f1"])
        prompt_toks.append(out.prompt_tokens)
        completion_toks.append(out.completion_tokens)

        rows.append(
            {
                "qid": ex.qid,
                "question": ex.question,
                "prediction": pred,
                "answers": ex.answers,
                "em": s["em"],
                "f1": s["f1"],
                "retrieved_ids": [r.item_id for r in out.retrieved],
            }
        )

    result["generation"] = {
        "num_samples": len(eval_examples),
        "EM": float(mean(em_values)) if em_values else 0.0,
        "F1": float(mean(f1_values)) if f1_values else 0.0,
        "em_values": em_values,
        "f1_values": f1_values,
        "latency_s": summarize_list(gen_lat),
        "prompt_tokens_mean": float(mean(prompt_toks)) if prompt_toks else 0.0,
        "completion_tokens_mean": float(mean(completion_toks)) if completion_toks else 0.0,
        "samples": rows,
    }

    return result


def benchmark_beir_retrieval(cfg: dict[str, Any]) -> dict[str, Any]:
    ds_cfg = cfg["datasets"]["beir"]
    max_q = int(ds_cfg["max_queries_per_dataset"])
    ks = [int(k) for k in cfg["benchmark"]["retrieval_ks"]]

    outputs: dict[str, Any] = {"task": "beir_retrieval", "datasets": {}, "macro_avg": {}}

    dataset_scores: list[dict[str, float]] = []

    for name in ds_cfg["names"]:
        docs, queries, qrels = load_beir_dataset(
            dataset_name=name,
            max_queries=max_q,
            seed=int(cfg["experiment"]["seed"]),
        )
        items = docs_to_items(docs)
        pipeline = RAGPipeline(cfg=cfg, items=items, enable_llm=False)

        runs, lat = _run_retrieval_loop(
            pipeline=pipeline,
            cfg=cfg,
            queries=queries,
            dedupe_to_doc_ids=True,
        )

        metrics, per_q = evaluate_retrieval(runs, qrels, ks=ks)
        outputs["datasets"][name] = {
            "num_docs": len(docs),
            "num_queries": len(queries),
            "metrics": metrics,
            "per_query": per_q,
            "latency_s": summarize_list(lat),
        }
        dataset_scores.append(metrics)

    if dataset_scores:
        keys = sorted({k for d in dataset_scores for k in d})
        outputs["macro_avg"] = {
            k: float(sum(d.get(k, 0.0) for d in dataset_scores) / len(dataset_scores)) for k in keys
        }

    return outputs
