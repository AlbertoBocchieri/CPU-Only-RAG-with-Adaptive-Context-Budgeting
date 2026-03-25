#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rag_cpu.chunking import chunk_documents
from rag_cpu.config import load_config
from rag_cpu.data import load_hotpotqa_distractor
from rag_cpu.pipeline import RAGPipeline
from rag_cpu.retrievers import chunks_to_items
from rag_cpu.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prewarm Hotpot full assets and persistent dense cache")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-queries", type=int, default=7405)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/prewarm_hotpot_full.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    cfg["retrieval"]["persist_embedding_cache"] = True

    examples, docs = load_hotpotqa_distractor(
        split=str(args.split),
        max_queries=int(args.num_queries),
        seed=int(args.seed),
    )
    t_data_s = time.perf_counter() - t0

    t_chunk0 = time.perf_counter()
    chunks, _ = chunk_documents(
        docs,
        chunk_size_words=int(cfg["chunking"]["chunk_size_words"]),
        chunk_overlap_words=int(cfg["chunking"]["chunk_overlap_words"]),
        min_chunk_words=int(cfg["chunking"]["min_chunk_words"]),
    )
    t_chunk_s = time.perf_counter() - t_chunk0

    t_index0 = time.perf_counter()
    _ = RAGPipeline(cfg=cfg, items=chunks_to_items(chunks), enable_llm=False)
    t_index_s = time.perf_counter() - t_index0

    cache_dir = Path(str(cfg["retrieval"].get("embedding_cache_dir", "cache/embeddings")))
    n_cache_files = len(list(cache_dir.glob("*.npy"))) if cache_dir.exists() else 0

    summary = {
        "config": str(args.config),
        "split": str(args.split),
        "num_queries": int(len(examples)),
        "num_docs": int(len(docs)),
        "num_chunks": int(len(chunks)),
        "embedding_cache_dir": str(cache_dir),
        "embedding_cache_files": int(n_cache_files),
        "timings_s": {
            "load_data": float(t_data_s),
            "chunking": float(t_chunk_s),
            "build_retrievers_with_dense_cache": float(t_index_s),
            "total": float(time.perf_counter() - t0),
        },
    }
    save_json(args.output, summary)
    print(f"Prewarm done. docs={len(docs)} chunks={len(chunks)}")
    print(f"Cache dir: {cache_dir} (files={n_cache_files})")
    print(f"Summary: {args.output}")


if __name__ == "__main__":
    main()
