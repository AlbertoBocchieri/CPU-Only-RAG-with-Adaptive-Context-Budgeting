#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rag_cpu.chunking import chunk_documents
from rag_cpu.config import load_config
from rag_cpu.ingest import load_local_documents
from rag_cpu.pipeline import RAGPipeline
from rag_cpu.retrievers import chunks_to_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over a local text corpus")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    docs = load_local_documents(args.corpus_dir)
    chunks, _ = chunk_documents(
        docs,
        chunk_size_words=int(cfg["chunking"]["chunk_size_words"]),
        chunk_overlap_words=int(cfg["chunking"]["chunk_overlap_words"]),
        min_chunk_words=int(cfg["chunking"]["min_chunk_words"]),
    )

    pipeline = RAGPipeline(cfg=cfg, items=chunks_to_items(chunks), enable_llm=True)
    out = pipeline.answer(args.question, cfg)

    print("\nAnswer:\n")
    print(out.answer or "")
    print("\nTop contexts:\n")
    for i, hit in enumerate(out.retrieved, start=1):
        snippet = (hit.text[:220] + "...") if len(hit.text) > 220 else hit.text
        print(f"[{i}] {hit.item_id} score={hit.score:.4f}\\n{snippet}\\n")


if __name__ == "__main__":
    main()
