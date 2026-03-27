# CPU-Only RAG on Multi-Hop QA

This is a **code-focused, GitHub-ready** version of the project.
It keeps only the source code and the minimum runtime assets needed to execute the pipeline correctly, while excluding paper artifacts, experimental results, query pools, and large local assets.

## What this repository contains

- `src/rag_cpu/`
  - legacy CPU-only RAG stack
  - benchmark runner, retrieval, generation, profiling, context budgeting
- `src/agnostic_cpu_rag/`
  - agnostic context controller and related runtime/evaluation utilities
- `scripts/`
  - main entrypoint: `scripts/benchmark_suite.py`
  - additional research and analysis scripts kept as source code
- `configs/`
  - `configs/base.yaml`: generic runnable baseline config
  - `configs/legacy_acbsc_eval/hotpot_agnostic_acb_sc_full7405_p4.yaml`: frozen ACB-SC example config
- `assets/`
  - two lightweight plots used in this README
- placeholder runtime directories:
  - `models/`
  - `data/`
  - `cache/`
  - `logs/`
  - `results/generated/`

## What is intentionally excluded

- paper sources, LaTeX figures, tables, and manuscript artifacts
- raw benchmark outputs and large result folders
- canonical query pools and full experiment manifests
- Windows handoff bundle
- downloaded model weights, caches, and datasets

If you want to reproduce the full paper campaign exactly, this bundle is not sufficient by design. It is meant to provide the runnable code and the minimal configs needed to execute the system.

## Pipeline overview

The system has five stages.

1. Corpus preparation
   - source documents are chunked into overlapping text spans
   - a shared retrieval corpus is built for the target dataset

2. Retrieval
   - BM25 sparse retrieval
   - sentence-transformer dense retrieval
   - optional hybrid fusion
   - optional multi-hop query expansion in best-quality settings

3. Context control
   - baseline mode passes the final retrieval set directly
   - ACB variants reduce the evidence set before generation
   - ACB-SC is the self-calibrating variant used in the frozen example config

4. Generation
   - `llama-cpp-python` with a local GGUF model
   - CPU-only inference

5. Evaluation
   - answer quality
   - retrieval quality
   - latency breakdown
   - optional power profiling on macOS

## Repository layout

```text
.
├── assets/
├── cache/
├── configs/
│   ├── base.yaml
│   └── legacy_acbsc_eval/
│       └── hotpot_agnostic_acb_sc_full7405_p4.yaml
├── data/
├── logs/
├── models/
├── results/
│   └── generated/
├── scripts/
├── src/
│   ├── agnostic_cpu_rag/
│   └── rag_cpu/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Representative plots

These plots summarize the main quality/latency behavior of the final systems.

### EM/F1 vs TTFT p50

![EM/F1 vs TTFT p50](assets/fig1_em_f1_vs_ttft.png)

### Latency breakdown

![Latency breakdown](assets/fig2_latency_breakdown.png)

## Requirements

- Python `3.11` or `3.12`
- a working C/C++ toolchain for `llama-cpp-python`
- enough RAM for a local 3B GGUF model
- internet access on first run for dataset and retriever downloads

On macOS, optional power profiling uses `powermetrics` and therefore requires `sudo`.

## Installation

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

## Required local asset

Place the GGUF model at:

```text
models/qwen2.5-3b-instruct-q4_k_m.gguf
```

The bundled configs expect exactly that path unless you override it in YAML.

## First-run downloads

This repository does **not** ship datasets or retriever weights. On first use it will download:

- Hugging Face datasets as needed
- sentence-transformer retrieval models
- optional cross-encoder reranker weights if a config enables reranking

These assets are cached under the default locations used by the relevant libraries plus the local `cache/` paths defined in the configs.

## Quick start

After activating the virtual environment, the main runnable entrypoint is:

```bash
PYTHONPATH=src python scripts/benchmark_suite.py --help
```

### Minimal baseline run

```bash
PYTHONPATH=src python scripts/benchmark_suite.py \
  --config configs/base.yaml \
  --dataset hotpot_qa \
  --tier A \
  --num-queries 20 \
  --run-id smoke_hotpot
```

### Minimal ACB-SC run

```bash
PYTHONPATH=src python scripts/benchmark_suite.py \
  --config configs/legacy_acbsc_eval/hotpot_agnostic_acb_sc_full7405_p4.yaml \
  --dataset hotpot_qa \
  --num-queries 20 \
  --run-id smoke_hotpot_acbsc
```

### Optional power profiling on macOS

```bash
sudo env PYTHONPATH=src python scripts/benchmark_suite.py \
  --config configs/legacy_acbsc_eval/hotpot_agnostic_acb_sc_full7405_p4.yaml \
  --dataset hotpot_qa \
  --num-queries 20 \
  --run-id smoke_hotpot_acbsc_power \
  --profile-power \
  --power-sampling-interval-ms 1000
```

## Output locations

- default outputs go under `results/`
- the repository keeps `results/generated/` tracked as a placeholder
- caches, logs, models, and generated outputs are ignored by `.gitignore`

## Notes on the included configs

- `configs/base.yaml`
  - generic runnable config
  - hybrid retrieval + reranker enabled
  - suitable for smoke tests and local validation

- `configs/legacy_acbsc_eval/hotpot_agnostic_acb_sc_full7405_p4.yaml`
  - frozen Hotpot ACB-SC example
  - weighted-sum hybrid retrieval + query expansion
  - self-calibrating context controller
  - 4-thread runtime profile

## Scope and limitations

- this bundle is optimized for code availability, not full-paper reproducibility
- many research scripts are included as source code, but some expect assets that are intentionally not bundled here
- the main supported path in this bundle is the benchmark/evaluation pipeline driven by `scripts/benchmark_suite.py`

## Sanity checks performed on this bundle

Before packaging, the bundle was checked for:

- clean Python imports for `rag_cpu` and `agnostic_cpu_rag`
- successful CLI startup for `scripts/benchmark_suite.py --help`
- static bytecode compilation of `src/` and `scripts/`

That means the repository structure is internally coherent; actual runs still require the local model and dataset downloads described above.
