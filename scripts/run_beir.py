#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rag_cpu.benchmark import benchmark_beir_retrieval
from rag_cpu.config import load_config
from rag_cpu.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BEIR retrieval benchmark")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--tuned-config", default="")
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--out-prefix", default="results/beir")
    args = parser.parse_args()

    base = load_config(args.base_config)
    tuned = load_config(args.tuned_config) if args.tuned_config else None

    def prep(cfg: dict) -> dict:
        c = copy.deepcopy(cfg)
        if args.disable_reranker:
            c["reranker"]["enabled"] = False
        if args.max_queries > 0:
            c["datasets"]["beir"]["max_queries_per_dataset"] = int(args.max_queries)
        return c

    base_res = benchmark_beir_retrieval(prep(base))
    save_json(f"{args.out_prefix}_baseline.json", base_res)
    print("Baseline macro:", base_res.get("macro_avg", {}))

    if tuned is not None:
        tuned_res = benchmark_beir_retrieval(prep(tuned))
        save_json(f"{args.out_prefix}_tuned.json", tuned_res)
        print("Tuned macro:", tuned_res.get("macro_avg", {}))


if __name__ == "__main__":
    main()
