#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from rag_cpu.config import deep_update, load_config, write_config

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs/base.yaml"
EVAL_CONFIG_DIR = ROOT / "configs/legacy_acbsc_eval"
WINDOWS_BUNDLE_DIR = ROOT / "windows/legacy_acbsc_runs"
HOTPOT_FULL_SOURCE_QIDS = ROOT / "results/full_7405_best_quality_base_power/cfg_a000fd2bb017/hotpot_qa/sampled_qids.json"
HOTPOT_POOL_DIR = ROOT / "results/task_family_weight_search/pools/hotpot_qa"
TWOWIKI_POOL_DIR = ROOT / "results/task_family_weight_search/pools/two_wiki_multihop"
SEED = 42
TWOWIKI_LARGE_SIZE = 6288

COMMON_RETRIEVAL = {
    "dense_model": "intfloat/e5-base-v2",
    "dense_prefix_query": "query: ",
    "dense_prefix_passage": "passage: ",
    "persist_embedding_cache": True,
    "embedding_cache_dir": "cache/embeddings",
    "retriever_mode": "hybrid",
    "fusion_method": "WEIGHTED_SUM",
    "rrf_k": 60,
    "weighted_alpha": 0.75,
    "top_k_dense": 150,
    "top_k_bm25": 250,
    "top_k_final": 20,
    "hybrid_alpha": 0.55,
    "normalize_scores": True,
    "agreement_bonus_enabled": False,
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

Q0_CONTEXT_BUDGETING = {"enabled": False}

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

HOTPOT_FULL_QIDS_OUT = HOTPOT_POOL_DIR / "full_7405_qids.json"
HOTPOT_FULL_META_OUT = HOTPOT_POOL_DIR / "full_7405_meta.json"
TWOWIKI_LARGE_QIDS_OUT = TWOWIKI_POOL_DIR / "large_holdout_6288_qids.json"
TWOWIKI_LARGE_META_OUT = TWOWIKI_POOL_DIR / "large_holdout_6288_meta.json"

HOTPOT_CONFIG_OUT = EVAL_CONFIG_DIR / "hotpot_agnostic_acb_sc_full7405_p4.yaml"
TWOWIKI_Q0_CONFIG_OUT = EVAL_CONFIG_DIR / "twowiki_q0_large6288_p4.yaml"
TWOWIKI_ACBSC_CONFIG_OUT = EVAL_CONFIG_DIR / "twowiki_agnostic_acb_sc_large6288_p4.yaml"

WINDOWS_CONFIG_DIR = WINDOWS_BUNDLE_DIR / "configs"
WINDOWS_QID_DIR = WINDOWS_BUNDLE_DIR / "qids"
WINDOWS_SRC_DIR = WINDOWS_BUNDLE_DIR / "src"
WINDOWS_RAG_CPU_DIR = WINDOWS_SRC_DIR / "rag_cpu"
WINDOWS_SCRIPT_DIR = WINDOWS_BUNDLE_DIR / "scripts"
WINDOWS_SCRIPT_LEGACY_DIR = WINDOWS_SCRIPT_DIR / "legacy_acbsc"
WINDOWS_RESULTS_DIR = WINDOWS_BUNDLE_DIR / "results"
WINDOWS_CACHE_DIR = WINDOWS_BUNDLE_DIR / "cache"
WINDOWS_MODELS_DIR = WINDOWS_BUNDLE_DIR / "models"
WINDOWS_COMMON_PS1 = WINDOWS_BUNDLE_DIR / "Common.ps1"
WINDOWS_README = WINDOWS_BUNDLE_DIR / "README_WINDOWS.md"
WINDOWS_HANDOFF = WINDOWS_BUNDLE_DIR / "CODEX_HANDOFF.md"
WINDOWS_PYPROJECT = WINDOWS_BUNDLE_DIR / "pyproject.toml"
WINDOWS_REQUIREMENTS = WINDOWS_BUNDLE_DIR / "requirements_windows_minimal.txt"
WINDOWS_HOTPOT_PS1 = WINDOWS_BUNDLE_DIR / "run_hotpot_full_acbsc.ps1"
WINDOWS_TWOWIKI_Q0_PS1 = WINDOWS_BUNDLE_DIR / "run_twowiki_large_q0.ps1"
WINDOWS_TWOWIKI_ACBSC_PS1 = WINDOWS_BUNDLE_DIR / "run_twowiki_large_acbsc.ps1"
WINDOWS_TWOWIKI_BOTH_PS1 = WINDOWS_BUNDLE_DIR / "run_twowiki_large_both.ps1"
WINDOWS_COMPARE_PS1 = WINDOWS_BUNDLE_DIR / "compare_twowiki_large.ps1"
WINDOWS_BENCH_SCRIPT = WINDOWS_SCRIPT_DIR / "benchmark_suite.py"
WINDOWS_COMPARE_SCRIPT = WINDOWS_SCRIPT_LEGACY_DIR / "compare_large_eval.py"
WINDOWS_BUILD_ASSETS_SCRIPT = WINDOWS_SCRIPT_LEGACY_DIR / "build_full_eval_assets.py"

SCRIPT_BENCHMARK_SUITE = ROOT / "scripts/benchmark_suite.py"
SCRIPT_COMPARE_LARGE = ROOT / "scripts/legacy_acbsc/compare_large_eval.py"
SCRIPT_BUILD_ASSETS = ROOT / "scripts/legacy_acbsc/build_full_eval_assets.py"
SRC_RAG_CPU_DIR = ROOT / "src/rag_cpu"
PYPROJECT_FILE = ROOT / "pyproject.toml"

USED_TWOWIKI_POOL_FILES = [
    TWOWIKI_POOL_DIR / "tuning_150_qids.json",
    TWOWIKI_POOL_DIR / "tuning_300_qids.json",
    TWOWIKI_POOL_DIR / "selection_pilot_75_qids.json",
    TWOWIKI_POOL_DIR / "representative_150_qids.json",
    TWOWIKI_POOL_DIR / "representative_300_qids.json",
    TWOWIKI_POOL_DIR / "holdout_1000_qids.json",
]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def load_twowiki_rows() -> list[dict[str, Any]]:
    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    rows: list[dict[str, Any]] = []
    for row in ds:
        qid = f"2wiki_{row['id']}"
        qtype = str(row.get("type", "")).strip() or "unknown"
        rows.append({"qid": qid, "type": qtype})
    return rows


def hamilton_targets(counts: dict[str, int], total_size: int) -> dict[str, int]:
    grand_total = sum(counts.values())
    exact = {label: (total_size * count / grand_total) for label, count in counts.items()}
    targets = {label: int(value) for label, value in exact.items()}
    remainder = total_size - sum(targets.values())
    ranked = sorted(exact, key=lambda label: (-(exact[label] - targets[label]), -counts[label], label))
    for label in ranked[:remainder]:
        targets[label] += 1
    return dict(sorted(targets.items()))


def rebalance_targets(targets: dict[str, int], available: dict[str, int], reference: dict[str, int], desired_total: int) -> dict[str, int]:
    out = {label: min(int(targets.get(label, 0)), int(available.get(label, 0))) for label in available}
    remaining = desired_total - sum(out.values())
    while remaining > 0:
        candidates = [
            label
            for label in sorted(available)
            if out[label] < int(available[label])
        ]
        if not candidates:
            break
        candidates.sort(key=lambda label: (-(available[label] - out[label]), -reference.get(label, 0), label))
        out[candidates[0]] += 1
        remaining -= 1
    if sum(out.values()) != desired_total:
        raise ValueError(f"Unable to allocate {desired_total} rows after capacity checks; got {sum(out.values())}")
    return dict(sorted(out.items()))


def build_large_twowiki_pool() -> dict[str, Any]:
    rows = load_twowiki_rows()
    full_counts = Counter(row["type"] for row in rows)
    used_qids: set[str] = set()
    used_sources: dict[str, int] = {}
    for path in USED_TWOWIKI_POOL_FILES:
        qids = json.loads(path.read_text(encoding="utf-8"))
        used_sources[str(path.relative_to(ROOT))] = len(qids)
        used_qids.update(str(qid) for qid in qids)

    candidates = [row for row in rows if row["qid"] not in used_qids]
    if len(candidates) < TWOWIKI_LARGE_SIZE:
        raise ValueError(f"Need {TWOWIKI_LARGE_SIZE} 2Wiki rows, only {len(candidates)} remain after exclusions")

    available_counts = Counter(row["type"] for row in candidates)
    target_counts = hamilton_targets(dict(full_counts), TWOWIKI_LARGE_SIZE)
    adjusted_targets = rebalance_targets(target_counts, dict(available_counts), dict(full_counts), TWOWIKI_LARGE_SIZE)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        buckets[row["type"]].append(row)
    for idx, label in enumerate(sorted(buckets)):
        rng = random.Random(SEED + idx + 1)
        rng.shuffle(buckets[label])

    selected_qids: set[str] = set()
    for label, count in adjusted_targets.items():
        bucket = buckets[label]
        if len(bucket) < count:
            raise ValueError(f"Bucket {label} too small: need {count}, have {len(bucket)}")
        for row in bucket[:count]:
            selected_qids.add(row["qid"])

    selected_rows = [row for row in rows if row["qid"] in selected_qids]
    if len(selected_rows) != TWOWIKI_LARGE_SIZE:
        raise ValueError(f"Selected {len(selected_rows)} rows instead of {TWOWIKI_LARGE_SIZE}")

    qids = [row["qid"] for row in selected_rows]
    actual_counts = Counter(row["type"] for row in selected_rows)
    atomic_write_json(TWOWIKI_LARGE_QIDS_OUT, qids)
    atomic_write_json(
        TWOWIKI_LARGE_META_OUT,
        {
            "dataset": "two_wiki_multihop",
            "pool_name": "large_holdout_6288",
            "size": len(qids),
            "seed": SEED,
            "qids_path": str(TWOWIKI_LARGE_QIDS_OUT),
            "distribution_target": dict(sorted(target_counts.items())),
            "distribution_actual": dict(sorted(actual_counts.items())),
            "distribution_available_after_exclusions": dict(sorted(available_counts.items())),
            "exclusions": {
                "used_pool_files": used_sources,
                "excluded_qids": len(used_qids),
            },
            "selection_policy": "proportional_hamilton_on_full_distribution_then_capacity_rebalanced_and_order_preserved",
        },
    )
    return {
        "qids": qids,
        "target_counts": dict(sorted(target_counts.items())),
        "actual_counts": dict(sorted(actual_counts.items())),
        "available_counts": dict(sorted(available_counts.items())),
        "excluded_qids": len(used_qids),
    }


def build_hotpot_full_pool() -> list[str]:
    qids = json.loads(HOTPOT_FULL_SOURCE_QIDS.read_text(encoding="utf-8"))
    qids = [str(qid) for qid in qids]
    if len(qids) != 7405:
        raise ValueError(f"Expected 7405 Hotpot qids, got {len(qids)}")
    atomic_write_json(HOTPOT_FULL_QIDS_OUT, qids)
    atomic_write_json(
        HOTPOT_FULL_META_OUT,
        {
            "dataset": "hotpot_qa",
            "pool_name": "full_7405",
            "size": len(qids),
            "source_qids_path": str(HOTPOT_FULL_SOURCE_QIDS),
            "qids_path": str(HOTPOT_FULL_QIDS_OUT),
            "selection_policy": "copied_from_existing_historical_full_hotpot_qids",
        },
    )
    return qids


def build_config(dataset_name: str, context_budgeting: dict[str, Any], *, max_new_tokens: int) -> dict[str, Any]:
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
            "llm": {"n_threads": 4, "temperature": 0.0, "max_new_tokens": max_new_tokens},
            "llm_runtime": dict(COMMON_RUNTIME),
            "context_budgeting": dict(context_budgeting),
        },
    )
    return cfg


def build_configs() -> None:
    EVAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    write_config(HOTPOT_CONFIG_OUT, build_config("hotpot_qa", AGNOSTIC_ACB_SC, max_new_tokens=64))
    write_config(TWOWIKI_Q0_CONFIG_OUT, build_config("two_wiki_multihop", Q0_CONTEXT_BUDGETING, max_new_tokens=32))
    write_config(TWOWIKI_ACBSC_CONFIG_OUT, build_config("two_wiki_multihop", AGNOSTIC_ACB_SC, max_new_tokens=32))


def windows_keep_awake_script() -> str:
    return r'''$ErrorActionPreference = "Stop"
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class PowerKeepAwake {
  [DllImport("kernel32.dll")]
  public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$ES_CONTINUOUS = 0x80000000
$ES_SYSTEM_REQUIRED = 0x00000001
$ES_DISPLAY_REQUIRED = 0x00000002
function Invoke-KeepAwake {
  param([scriptblock]$Script)
  [PowerKeepAwake]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED) | Out-Null
  try {
    & $Script
  }
  finally {
    [PowerKeepAwake]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
  }
}
'''


def windows_launcher(script_name: str, command: str) -> str:
    return f'''$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "Common.ps1")
$env:PYTHONPATH = "src"
Invoke-KeepAwake {{
{command}
}}
'''


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_filtered(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def requirements_text() -> str:
    return "\n".join(
        [
            "numpy>=1.26,<3",
            "pandas>=2.2,<3",
            "pyyaml>=6.0",
            "tqdm>=4.66",
            "scipy>=1.12",
            "scikit-learn>=1.4",
            "datasets>=2.18",
            "sentence-transformers>=3.0",
            "rank-bm25>=0.2.2",
            "llama-cpp-python>=0.3.7",
            "beir>=2.0.0",
            "tabulate>=0.9.0",
            "psutil>=5.9",
            "matplotlib>=3.8,<4",
            "rich>=13.7,<15",
            "",
        ]
    )


def build_windows_bundle() -> None:
    reset_dir(WINDOWS_CONFIG_DIR)
    reset_dir(WINDOWS_QID_DIR)
    reset_dir(WINDOWS_SRC_DIR)
    reset_dir(WINDOWS_SCRIPT_DIR)
    WINDOWS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_SCRIPT_LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HOTPOT_CONFIG_OUT, WINDOWS_CONFIG_DIR / HOTPOT_CONFIG_OUT.name)
    shutil.copy2(TWOWIKI_Q0_CONFIG_OUT, WINDOWS_CONFIG_DIR / TWOWIKI_Q0_CONFIG_OUT.name)
    shutil.copy2(TWOWIKI_ACBSC_CONFIG_OUT, WINDOWS_CONFIG_DIR / TWOWIKI_ACBSC_CONFIG_OUT.name)
    shutil.copy2(HOTPOT_FULL_QIDS_OUT, WINDOWS_QID_DIR / HOTPOT_FULL_QIDS_OUT.name)
    shutil.copy2(TWOWIKI_LARGE_QIDS_OUT, WINDOWS_QID_DIR / TWOWIKI_LARGE_QIDS_OUT.name)
    shutil.copy2(PYPROJECT_FILE, WINDOWS_PYPROJECT)
    shutil.copy2(SCRIPT_BENCHMARK_SUITE, WINDOWS_BENCH_SCRIPT)
    shutil.copy2(SCRIPT_COMPARE_LARGE, WINDOWS_COMPARE_SCRIPT)
    shutil.copy2(SCRIPT_BUILD_ASSETS, WINDOWS_BUILD_ASSETS_SCRIPT)
    copy_tree_filtered(SRC_RAG_CPU_DIR, WINDOWS_RAG_CPU_DIR)
    atomic_write_text(WINDOWS_REQUIREMENTS, requirements_text())

    atomic_write_text(WINDOWS_COMMON_PS1, windows_keep_awake_script())
    atomic_write_text(
        WINDOWS_HOTPOT_PS1,
        windows_launcher(
            "run_hotpot_full_acbsc.ps1",
            '  & ".venv\\Scripts\\python.exe" "scripts\\benchmark_suite.py" --config "configs\\hotpot_agnostic_acb_sc_full7405_p4.yaml" --dataset hotpot_qa --num-queries 7405 --query-ids-path "qids\\full_7405_qids.json" --seed 42 --run-id legacy_acbsc_hotpot_full7405 --output-dir "results\\legacy_acbsc_full_hotpot" --ui-update-every 1',
        ),
    )
    atomic_write_text(
        WINDOWS_TWOWIKI_Q0_PS1,
        windows_launcher(
            "run_twowiki_large_q0.ps1",
            '  & ".venv\\Scripts\\python.exe" "scripts\\benchmark_suite.py" --config "configs\\twowiki_q0_large6288_p4.yaml" --dataset two_wiki_multihop --num-queries 6288 --query-ids-path "qids\\large_holdout_6288_qids.json" --seed 42 --run-id legacy_q0_twowiki_large6288 --output-dir "results\\legacy_acbsc_large_twowiki" --ui-update-every 1',
        ),
    )
    atomic_write_text(
        WINDOWS_TWOWIKI_ACBSC_PS1,
        windows_launcher(
            "run_twowiki_large_acbsc.ps1",
            '  & ".venv\\Scripts\\python.exe" "scripts\\benchmark_suite.py" --config "configs\\twowiki_agnostic_acb_sc_large6288_p4.yaml" --dataset two_wiki_multihop --num-queries 6288 --query-ids-path "qids\\large_holdout_6288_qids.json" --seed 42 --run-id legacy_acbsc_twowiki_large6288 --output-dir "results\\legacy_acbsc_large_twowiki" --ui-update-every 1',
        ),
    )
    atomic_write_text(
        WINDOWS_TWOWIKI_BOTH_PS1,
        windows_launcher(
            "run_twowiki_large_both.ps1",
            '  & ".venv\\Scripts\\python.exe" "scripts\\benchmark_suite.py" --config "configs\\twowiki_q0_large6288_p4.yaml" --config-2 "configs\\twowiki_agnostic_acb_sc_large6288_p4.yaml" --dataset two_wiki_multihop --num-queries 6288 --query-ids-path "qids\\large_holdout_6288_qids.json" --seed 42 --run-id legacy_twowiki_large6288_q0_vs_acbsc --output-dir "results\\legacy_acbsc_large_twowiki" --ui-update-every 1',
        ),
    )
    atomic_write_text(
        WINDOWS_COMPARE_PS1,
        windows_launcher(
            "compare_twowiki_large.ps1",
            '  & ".venv\\Scripts\\python.exe" "scripts\\legacy_acbsc\\compare_large_eval.py" --baseline-run-root "results\\legacy_acbsc_large_twowiki\\legacy_q0_twowiki_large6288" --candidate-run-root "results\\legacy_acbsc_large_twowiki\\legacy_acbsc_twowiki_large6288" --dataset two_wiki_multihop --baseline-label legacy_q0_query_expansion --candidate-label new_agnostic_acb_sc --output-dir "results\\legacy_acbsc_large_twowiki\\compare" --output-name twowiki_legacy_acbsc_vs_q0_large6288_compare.json',
        ),
    )

    atomic_write_text(
        WINDOWS_README,
        """# Windows Legacy ACB-SC Run Bundle\n\n## Purpose\nThis bundle prepares the legacy-stack runs for:\n- Hotpot full 7405 with `agnostic_acb_sc`\n- 2Wiki large 6288 baseline `Q0` with query expansion\n- 2Wiki large 6288 `agnostic_acb_sc` on the same qids\n\n## Standalone Bundle Contract\nThis folder is meant to be copied to Windows **without** copying the whole repo.\nOnly two things are expected to be recreated separately on Windows:\n- a local Python virtual environment in `.venv\\`\n- the GGUF model file in `models\\qwen2.5-3b-instruct-q4_k_m.gguf`\n\nEverything else strictly needed for these runs is bundled here.\n\n## What Is Included\n- Frozen eval configs in `configs\\`\n- Frozen qid files in `qids\\`\n- PowerShell launchers in the bundle root\n- Python entrypoints in `scripts\\`\n- Full `rag_cpu` source package in `src\\rag_cpu\\`\n- `pyproject.toml` and `requirements_windows_minimal.txt` to recreate the environment\n- Empty working directories: `results\\`, `cache\\`, `models\\`\n\n## Bundle Structure\n- Bundle root:\n  - `.venv\\Scripts\\python.exe`: interpreter expected by launchers after you create the env\n  - `models\\`: place the GGUF model here\n  - `results\\`: run artifacts are written here\n  - `cache\\`: embedding and dataset caches can live here\n- Internal files:\n  - `configs\\`: frozen YAML configs for the exact runs\n  - `qids\\`: frozen query-id files\n  - `scripts\\benchmark_suite.py`: benchmark entrypoint used by the launchers\n  - `scripts\\legacy_acbsc\\compare_large_eval.py`: compare entrypoint for 2Wiki\n  - `src\\rag_cpu\\`: legacy source package used at runtime\n  - `run_*.ps1`: PowerShell launchers\n  - `README_WINDOWS.md`: operator instructions\n  - `CODEX_HANDOFF.md`: Codex-oriented handoff and constraints\n  - `requirements_windows_minimal.txt`: minimal dependency list for the Python env\n\n## Setup On Windows\n1. Copy the whole `legacy_acbsc_runs` folder to Windows.\n2. Open PowerShell inside the bundle root.\n3. Create the virtual environment:\n   - `py -3.11 -m venv .venv`\n4. Activate it:\n   - `.\\.venv\\Scripts\\Activate.ps1`\n5. Install dependencies:\n   - `python -m pip install --upgrade pip`\n   - `python -m pip install -r requirements_windows_minimal.txt`\n6. Place the model at:\n   - `models\\qwen2.5-3b-instruct-q4_k_m.gguf`\n\n## Invariants\n- Legacy `rag_cpu` stack only\n- Threads fixed to `4/4`\n- Hotpot: only ACB-SC full run is launched here\n- 2Wiki baseline is `Q0` with query expansion, not `incremental_sc`\n- Do not resample qids\n- Do not retune configs before running\n\n## Run Procedure\n1. Open PowerShell in the bundle root.\n2. Verify `.venv\\Scripts\\python.exe` exists.\n3. Launch in this order:\n   - `powershell -ExecutionPolicy Bypass -File .\\run_hotpot_full_acbsc.ps1`\n   - `powershell -ExecutionPolicy Bypass -File .\\run_twowiki_large_q0.ps1`\n   - `powershell -ExecutionPolicy Bypass -File .\\run_twowiki_large_acbsc.ps1`\n   - `powershell -ExecutionPolicy Bypass -File .\\compare_twowiki_large.ps1`\n4. Inspect the output directories listed below.\n\nOr use `run_twowiki_large_both.ps1` for the two 2Wiki runs together.\n\n## Outputs\n- Hotpot full run:\n  - `results\\legacy_acbsc_full_hotpot\\legacy_acbsc_hotpot_full7405\\`\n- 2Wiki large baseline:\n  - `results\\legacy_acbsc_large_twowiki\\legacy_q0_twowiki_large6288\\`\n- 2Wiki large ACB-SC:\n  - `results\\legacy_acbsc_large_twowiki\\legacy_acbsc_twowiki_large6288\\`\n- 2Wiki compare:\n  - `results\\legacy_acbsc_large_twowiki\\compare\\twowiki_legacy_acbsc_vs_q0_large6288_compare.json`\n""",
    )

    atomic_write_text(
        WINDOWS_HANDOFF,
        """# Codex Handoff: Legacy Full Evaluation Bundle\n\n## Goal\nRun the next legacy-stack evaluation stage without changing the retrieval pipeline, prompt, model, or runtime. The only candidate controller under test is `agnostic_acb_sc`.\n\n## Standalone Assumption\nThis bundle is intended to run on Windows **without the rest of the repo**.\nOnly these two items are expected to be recreated manually:\n- `.venv\\` Python environment\n- `models\\qwen2.5-3b-instruct-q4_k_m.gguf`\n\nEverything else needed by this stage is bundled here.\n\n## Project Structure\n- Bundle root:\n  - `configs/`: frozen YAML configs\n  - `qids/`: frozen qid files\n  - `scripts/benchmark_suite.py`: benchmark entrypoint\n  - `scripts/legacy_acbsc/compare_large_eval.py`: compare entrypoint\n  - `src/rag_cpu/`: runtime source package\n  - `results/`: output root created/used by the launchers\n  - `cache/`: local cache root\n  - `models/`: place the GGUF model here\n  - `requirements_windows_minimal.txt`: dependency list for the env\n  - `pyproject.toml`: original dependency metadata snapshot\n  - `run_*.ps1`: launchers\n\n## Frozen Assets\n- Hotpot qids: `qids/full_7405_qids.json`\n- 2Wiki qids: `qids/large_holdout_6288_qids.json`\n- Hotpot config: `configs/hotpot_agnostic_acb_sc_full7405_p4.yaml`\n- 2Wiki Q0 config: `configs/twowiki_q0_large6288_p4.yaml`\n- 2Wiki ACB-SC config: `configs/twowiki_agnostic_acb_sc_large6288_p4.yaml`\n\n## Included Source Files\n- Benchmark entrypoint: `scripts/benchmark_suite.py`\n- 2Wiki compare entrypoint: `scripts/legacy_acbsc/compare_large_eval.py`\n- Asset builder snapshot: `scripts/legacy_acbsc/build_full_eval_assets.py`\n- Full legacy source package: `src/rag_cpu/`\n\nThe launchers set `PYTHONPATH=src` and run entirely from the bundle root.\n\n## Setup On Windows\n1. Copy the bundle to the target machine.\n2. Open PowerShell inside the bundle root.\n3. Create env:\n   - `py -3.11 -m venv .venv`\n4. Activate env:\n   - `.\\.venv\\Scripts\\Activate.ps1`\n5. Install deps:\n   - `python -m pip install --upgrade pip`\n   - `python -m pip install -r requirements_windows_minimal.txt`\n6. Place the model file at:\n   - `models\\qwen2.5-3b-instruct-q4_k_m.gguf`\n\n## Run Intent\n- Hotpot full: generate the ACB-SC full result on the historical 7405 qids\n- 2Wiki large baseline: Q0 with query expansion and no context budgeting\n- 2Wiki large candidate: ACB-SC on the same exact 6288 qids and same order\n\n## Exact Commands\n- Hotpot full:\n  - `powershell -ExecutionPolicy Bypass -File .\\run_hotpot_full_acbsc.ps1`\n- 2Wiki Q0:\n  - `powershell -ExecutionPolicy Bypass -File .\\run_twowiki_large_q0.ps1`\n- 2Wiki ACB-SC:\n  - `powershell -ExecutionPolicy Bypass -File .\\run_twowiki_large_acbsc.ps1`\n- 2Wiki compare:\n  - `powershell -ExecutionPolicy Bypass -File .\\compare_twowiki_large.ps1`\n\n## Run Order\n1. Run Hotpot full ACB-SC.\n2. Run 2Wiki Q0 baseline.\n3. Run 2Wiki ACB-SC on the same qids.\n4. Run the 2Wiki compare step.\n5. Do not alter configs or qids between steps.\n\n## Non-Negotiable Constraints\n- Keep threads at `4/4`\n- Do not alter qid files\n- Do not change model path, prompt template, or retrieval settings\n- Do not replace the 2Wiki Q0 baseline with `incremental_sc`\n- Do not retune ACB-SC before running\n\n## Expected Outputs\n- `results/legacy_acbsc_full_hotpot/legacy_acbsc_hotpot_full7405/`\n- `results/legacy_acbsc_large_twowiki/legacy_q0_twowiki_large6288/`\n- `results/legacy_acbsc_large_twowiki/legacy_acbsc_twowiki_large6288/`\n- `results/legacy_acbsc_large_twowiki/compare/twowiki_legacy_acbsc_vs_q0_large6288_compare.json`\n\n## Interpretation\n- Hotpot full is a standalone ACB-SC run artifact\n- 2Wiki large is the paired controller-level comparison against Q0\n- The 2Wiki compare artifact is the authoritative summary for baseline-vs-ACB-SC on the large pool\n""",
    )


def main() -> None:
    hotpot_qids = build_hotpot_full_pool()
    twowiki_info = build_large_twowiki_pool()
    build_configs()
    build_windows_bundle()
    print(
        json.dumps(
            {
                "hotpot_full": {
                    "count": len(hotpot_qids),
                    "qids_path": str(HOTPOT_FULL_QIDS_OUT.relative_to(ROOT)),
                    "config_path": str(HOTPOT_CONFIG_OUT.relative_to(ROOT)),
                },
                "twowiki_large": {
                    "count": len(twowiki_info["qids"]),
                    "qids_path": str(TWOWIKI_LARGE_QIDS_OUT.relative_to(ROOT)),
                    "target_counts": twowiki_info["target_counts"],
                    "actual_counts": twowiki_info["actual_counts"],
                    "excluded_qids": twowiki_info["excluded_qids"],
                    "q0_config_path": str(TWOWIKI_Q0_CONFIG_OUT.relative_to(ROOT)),
                    "acbsc_config_path": str(TWOWIKI_ACBSC_CONFIG_OUT.relative_to(ROOT)),
                },
                "windows_bundle": str(WINDOWS_BUNDLE_DIR.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
