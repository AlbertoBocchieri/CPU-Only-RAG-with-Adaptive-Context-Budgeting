#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY=".venv/bin/python"
export PYTHONPATH=src

OUT_ROOT="results/final_acb_sc"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

HOTPOT_QIDS="$OUT_ROOT/hotpot_prefix1000_qids.json"
TWOWIKI_QIDS="$OUT_ROOT/twowiki50_q1sc_canonical_qids.json"

NEW_CFG="configs/agnostic_cpu_rag/fixed_alpha_nohop2_multihop_stable.yaml"
OLD_HOTPOT_CFG="configs/retrieval_iters/e5_ws_multihop_quality_acb_sc_relaxed_seed4_p4.yaml"
OLD_TWOWIKI_CFG="configs/retrieval_iters/e5_ws_twowiki_quality_acb_sc_p4.yaml"
CAFFEINATE="caffeinate -di"

run_if_missing() {
  local sentinel="$1"
  shift
  if [[ -e "$sentinel" ]]; then
    echo "[skip] $sentinel"
    return 0
  fi
  echo "[run] $*"
  "$@"
}

resolve_legacy_run_dir() {
  local run_root="$1"
  local dataset="$2"
  find "$run_root" -path "*/$dataset/summary.json" -print | head -n 1 | xargs -I{} dirname "{}"
}

# P3: new system, Hotpot 1000, EWMA
NEW_HOTPOT_EWMA_RUN="agnostic_hotpot1000_multihop_stable_ewma"
NEW_HOTPOT_EWMA_DIR="$OUT_ROOT/$NEW_HOTPOT_EWMA_RUN/hotpot_qa"
run_if_missing \
  "$NEW_HOTPOT_EWMA_DIR/summary.json" \
  $CAFFEINATE "$PY" scripts/agnostic_cpu_rag/run_benchmark.py \
    --config "$NEW_CFG" \
    --dataset hotpot_qa \
    --num-queries 1000 \
    --include-qids-path "$HOTPOT_QIDS" \
    --run-id "$NEW_HOTPOT_EWMA_RUN" \
    --output-dir "$OUT_ROOT"

run_if_missing \
  "$OUT_ROOT/reports/agnostic_hotpot1000_ewma_runtime.json" \
  "$PY" scripts/agnostic_cpu_rag/run_extras_report.py \
    "$NEW_HOTPOT_EWMA_DIR/per_query.jsonl" \
    --summary-json "$NEW_HOTPOT_EWMA_DIR/summary.json" \
    --output "$OUT_ROOT/reports/agnostic_hotpot1000_ewma_runtime.json"

run_if_missing \
  "$OUT_ROOT/overrides/agnostic_hotpot1000_fixed_cap.yaml" \
  "$PY" scripts/agnostic_cpu_rag/derive_fixed_cap_override.py \
    "$OUT_ROOT/reports/agnostic_hotpot1000_ewma_runtime.json" \
    --system agnostic \
    --output "$OUT_ROOT/overrides/agnostic_hotpot1000_fixed_cap.yaml"

# P3 fixed-cap
NEW_HOTPOT_FIXED_RUN="agnostic_hotpot1000_multihop_stable_fixedcap"
NEW_HOTPOT_FIXED_DIR="$OUT_ROOT/$NEW_HOTPOT_FIXED_RUN/hotpot_qa"
run_if_missing \
  "$NEW_HOTPOT_FIXED_DIR/summary.json" \
  $CAFFEINATE "$PY" scripts/agnostic_cpu_rag/run_benchmark.py \
    --config "$NEW_CFG" \
    --config-override "$OUT_ROOT/overrides/agnostic_hotpot1000_fixed_cap.yaml" \
    --dataset hotpot_qa \
    --num-queries 1000 \
    --include-qids-path "$HOTPOT_QIDS" \
    --run-id "$NEW_HOTPOT_FIXED_RUN" \
    --output-dir "$OUT_ROOT"

run_if_missing \
  "$OUT_ROOT/reports/agnostic_hotpot1000_fixed_runtime.json" \
  "$PY" scripts/agnostic_cpu_rag/run_extras_report.py \
    "$NEW_HOTPOT_FIXED_DIR/per_query.jsonl" \
    --summary-json "$NEW_HOTPOT_FIXED_DIR/summary.json" \
    --output "$OUT_ROOT/reports/agnostic_hotpot1000_fixed_runtime.json"

run_if_missing \
  "$OUT_ROOT/compares/agnostic_hotpot1000_prefill_compare.json" \
  "$PY" scripts/agnostic_cpu_rag/compare_prefill_tracking.py \
    "$OUT_ROOT/reports/agnostic_hotpot1000_ewma_runtime.json" \
    "$OUT_ROOT/reports/agnostic_hotpot1000_fixed_runtime.json" \
    --output "$OUT_ROOT/compares/agnostic_hotpot1000_prefill_compare.json"

run_if_missing \
  "$OUT_ROOT/bootstrap/agnostic_hotpot1000_ewma_vs_fixed_f1_ci.json" \
  bash -lc 'PYTHONPATH=src .venv/bin/python scripts/agnostic_cpu_rag/paired_bootstrap_ci.py \
    "$0" "$1" --metric-path metrics.f1 > "$2"' \
  "$NEW_HOTPOT_FIXED_DIR/per_query.jsonl" \
  "$NEW_HOTPOT_EWMA_DIR/per_query.jsonl" \
  "$OUT_ROOT/bootstrap/agnostic_hotpot1000_ewma_vs_fixed_f1_ci.json"

# P5: legacy system, Hotpot 1000, EWMA
LEGACY_HOTPOT_EWMA_RUN="legacy_hotpot1000_q1sc_ewma"
run_if_missing \
  "$OUT_ROOT/$LEGACY_HOTPOT_EWMA_RUN" \
  $CAFFEINATE "$PY" scripts/benchmark_suite.py \
    --config "$OLD_HOTPOT_CFG" \
    --dataset hotpot_qa \
    --num-queries 1000 \
    --query-ids-path "$HOTPOT_QIDS" \
    --run-id "$LEGACY_HOTPOT_EWMA_RUN" \
    --output-dir "$OUT_ROOT"
LEGACY_HOTPOT_EWMA_DIR="$(resolve_legacy_run_dir "$OUT_ROOT/$LEGACY_HOTPOT_EWMA_RUN" hotpot_qa)"

run_if_missing \
  "$OUT_ROOT/reports/legacy_hotpot1000_ewma_runtime.json" \
  "$PY" scripts/agnostic_cpu_rag/run_extras_report.py \
    "$LEGACY_HOTPOT_EWMA_DIR/per_query.jsonl" \
    --summary-json "$LEGACY_HOTPOT_EWMA_DIR/summary.json" \
    --output "$OUT_ROOT/reports/legacy_hotpot1000_ewma_runtime.json"

run_if_missing \
  "$OUT_ROOT/overrides/legacy_hotpot1000_fixed_cap.yaml" \
  "$PY" scripts/agnostic_cpu_rag/derive_fixed_cap_override.py \
    "$OUT_ROOT/reports/legacy_hotpot1000_ewma_runtime.json" \
    --system legacy \
    --output "$OUT_ROOT/overrides/legacy_hotpot1000_fixed_cap.yaml"

LEGACY_HOTPOT_FIXED_RUN="legacy_hotpot1000_q1sc_fixedcap"
run_if_missing \
  "$OUT_ROOT/$LEGACY_HOTPOT_FIXED_RUN" \
  $CAFFEINATE "$PY" scripts/benchmark_suite.py \
    --config "$OLD_HOTPOT_CFG" \
    --config-override "$OUT_ROOT/overrides/legacy_hotpot1000_fixed_cap.yaml" \
    --dataset hotpot_qa \
    --num-queries 1000 \
    --query-ids-path "$HOTPOT_QIDS" \
    --run-id "$LEGACY_HOTPOT_FIXED_RUN" \
    --output-dir "$OUT_ROOT"
LEGACY_HOTPOT_FIXED_DIR="$(resolve_legacy_run_dir "$OUT_ROOT/$LEGACY_HOTPOT_FIXED_RUN" hotpot_qa)"

run_if_missing \
  "$OUT_ROOT/reports/legacy_hotpot1000_fixed_runtime.json" \
  "$PY" scripts/agnostic_cpu_rag/run_extras_report.py \
    "$LEGACY_HOTPOT_FIXED_DIR/per_query.jsonl" \
    --summary-json "$LEGACY_HOTPOT_FIXED_DIR/summary.json" \
    --output "$OUT_ROOT/reports/legacy_hotpot1000_fixed_runtime.json"

run_if_missing \
  "$OUT_ROOT/compares/legacy_hotpot1000_prefill_compare.json" \
  "$PY" scripts/agnostic_cpu_rag/compare_prefill_tracking.py \
    "$OUT_ROOT/reports/legacy_hotpot1000_ewma_runtime.json" \
    "$OUT_ROOT/reports/legacy_hotpot1000_fixed_runtime.json" \
    --output "$OUT_ROOT/compares/legacy_hotpot1000_prefill_compare.json"

run_if_missing \
  "$OUT_ROOT/bootstrap/legacy_hotpot1000_ewma_vs_fixed_f1_ci.json" \
  bash -lc 'PYTHONPATH=src .venv/bin/python scripts/agnostic_cpu_rag/paired_bootstrap_ci.py \
    "$0" "$1" --metric-path answer_metrics_per_query.f1 > "$2"' \
  "$LEGACY_HOTPOT_FIXED_DIR/per_query.jsonl" \
  "$LEGACY_HOTPOT_EWMA_DIR/per_query.jsonl" \
  "$OUT_ROOT/bootstrap/legacy_hotpot1000_ewma_vs_fixed_f1_ci.json"

# P4: new system, 2Wiki 50
NEW_TWOWIKI_EWMA_RUN="agnostic_twowiki50_multihop_stable_ewma"
NEW_TWOWIKI_EWMA_DIR="$OUT_ROOT/$NEW_TWOWIKI_EWMA_RUN/two_wiki_multihop"
run_if_missing \
  "$NEW_TWOWIKI_EWMA_DIR/summary.json" \
  $CAFFEINATE "$PY" scripts/agnostic_cpu_rag/run_benchmark.py \
    --config "$NEW_CFG" \
    --dataset two_wiki_multihop \
    --num-queries 50 \
    --include-qids-path "$TWOWIKI_QIDS" \
    --run-id "$NEW_TWOWIKI_EWMA_RUN" \
    --output-dir "$OUT_ROOT"

run_if_missing \
  "$OUT_ROOT/reports/agnostic_twowiki50_ewma_runtime.json" \
  "$PY" scripts/agnostic_cpu_rag/run_extras_report.py \
    "$NEW_TWOWIKI_EWMA_DIR/per_query.jsonl" \
    --summary-json "$NEW_TWOWIKI_EWMA_DIR/summary.json" \
    --output "$OUT_ROOT/reports/agnostic_twowiki50_ewma_runtime.json"

run_if_missing \
  "$OUT_ROOT/overrides/agnostic_twowiki50_fixed_cap.yaml" \
  "$PY" scripts/agnostic_cpu_rag/derive_fixed_cap_override.py \
    "$OUT_ROOT/reports/agnostic_twowiki50_ewma_runtime.json" \
    --system agnostic \
    --output "$OUT_ROOT/overrides/agnostic_twowiki50_fixed_cap.yaml"

NEW_TWOWIKI_FIXED_RUN="agnostic_twowiki50_multihop_stable_fixedcap"
NEW_TWOWIKI_FIXED_DIR="$OUT_ROOT/$NEW_TWOWIKI_FIXED_RUN/two_wiki_multihop"
run_if_missing \
  "$NEW_TWOWIKI_FIXED_DIR/summary.json" \
  $CAFFEINATE "$PY" scripts/agnostic_cpu_rag/run_benchmark.py \
    --config "$NEW_CFG" \
    --config-override "$OUT_ROOT/overrides/agnostic_twowiki50_fixed_cap.yaml" \
    --dataset two_wiki_multihop \
    --num-queries 50 \
    --include-qids-path "$TWOWIKI_QIDS" \
    --run-id "$NEW_TWOWIKI_FIXED_RUN" \
    --output-dir "$OUT_ROOT"

run_if_missing \
  "$OUT_ROOT/reports/agnostic_twowiki50_fixed_runtime.json" \
  "$PY" scripts/agnostic_cpu_rag/run_extras_report.py \
    "$NEW_TWOWIKI_FIXED_DIR/per_query.jsonl" \
    --summary-json "$NEW_TWOWIKI_FIXED_DIR/summary.json" \
    --output "$OUT_ROOT/reports/agnostic_twowiki50_fixed_runtime.json"

run_if_missing \
  "$OUT_ROOT/compares/agnostic_twowiki50_prefill_compare.json" \
  "$PY" scripts/agnostic_cpu_rag/compare_prefill_tracking.py \
    "$OUT_ROOT/reports/agnostic_twowiki50_ewma_runtime.json" \
    "$OUT_ROOT/reports/agnostic_twowiki50_fixed_runtime.json" \
    --output "$OUT_ROOT/compares/agnostic_twowiki50_prefill_compare.json"

run_if_missing \
  "$OUT_ROOT/bootstrap/agnostic_twowiki50_ewma_vs_fixed_f1_ci.json" \
  bash -lc 'PYTHONPATH=src .venv/bin/python scripts/agnostic_cpu_rag/paired_bootstrap_ci.py \
    "$0" "$1" --metric-path metrics.f1 > "$2"' \
  "$NEW_TWOWIKI_FIXED_DIR/per_query.jsonl" \
  "$NEW_TWOWIKI_EWMA_DIR/per_query.jsonl" \
  "$OUT_ROOT/bootstrap/agnostic_twowiki50_ewma_vs_fixed_f1_ci.json"

# P6: legacy system, 2Wiki 50
LEGACY_TWOWIKI_EWMA_RUN="legacy_twowiki50_q1sc_ewma"
run_if_missing \
  "$OUT_ROOT/$LEGACY_TWOWIKI_EWMA_RUN" \
  $CAFFEINATE "$PY" scripts/benchmark_suite.py \
    --config "$OLD_TWOWIKI_CFG" \
    --dataset two_wiki_multihop \
    --num-queries 50 \
    --query-ids-path "$TWOWIKI_QIDS" \
    --run-id "$LEGACY_TWOWIKI_EWMA_RUN" \
    --output-dir "$OUT_ROOT"
LEGACY_TWOWIKI_EWMA_DIR="$(resolve_legacy_run_dir "$OUT_ROOT/$LEGACY_TWOWIKI_EWMA_RUN" two_wiki_multihop)"

run_if_missing \
  "$OUT_ROOT/reports/legacy_twowiki50_ewma_runtime.json" \
  "$PY" scripts/agnostic_cpu_rag/run_extras_report.py \
    "$LEGACY_TWOWIKI_EWMA_DIR/per_query.jsonl" \
    --summary-json "$LEGACY_TWOWIKI_EWMA_DIR/summary.json" \
    --output "$OUT_ROOT/reports/legacy_twowiki50_ewma_runtime.json"

run_if_missing \
  "$OUT_ROOT/overrides/legacy_twowiki50_fixed_cap.yaml" \
  "$PY" scripts/agnostic_cpu_rag/derive_fixed_cap_override.py \
    "$OUT_ROOT/reports/legacy_twowiki50_ewma_runtime.json" \
    --system legacy \
    --output "$OUT_ROOT/overrides/legacy_twowiki50_fixed_cap.yaml"

LEGACY_TWOWIKI_FIXED_RUN="legacy_twowiki50_q1sc_fixedcap"
run_if_missing \
  "$OUT_ROOT/$LEGACY_TWOWIKI_FIXED_RUN" \
  $CAFFEINATE "$PY" scripts/benchmark_suite.py \
    --config "$OLD_TWOWIKI_CFG" \
    --config-override "$OUT_ROOT/overrides/legacy_twowiki50_fixed_cap.yaml" \
    --dataset two_wiki_multihop \
    --num-queries 50 \
    --query-ids-path "$TWOWIKI_QIDS" \
    --run-id "$LEGACY_TWOWIKI_FIXED_RUN" \
    --output-dir "$OUT_ROOT"
LEGACY_TWOWIKI_FIXED_DIR="$(resolve_legacy_run_dir "$OUT_ROOT/$LEGACY_TWOWIKI_FIXED_RUN" two_wiki_multihop)"

run_if_missing \
  "$OUT_ROOT/reports/legacy_twowiki50_fixed_runtime.json" \
  "$PY" scripts/agnostic_cpu_rag/run_extras_report.py \
    "$LEGACY_TWOWIKI_FIXED_DIR/per_query.jsonl" \
    --summary-json "$LEGACY_TWOWIKI_FIXED_DIR/summary.json" \
    --output "$OUT_ROOT/reports/legacy_twowiki50_fixed_runtime.json"

run_if_missing \
  "$OUT_ROOT/compares/legacy_twowiki50_prefill_compare.json" \
  "$PY" scripts/agnostic_cpu_rag/compare_prefill_tracking.py \
    "$OUT_ROOT/reports/legacy_twowiki50_ewma_runtime.json" \
    "$OUT_ROOT/reports/legacy_twowiki50_fixed_runtime.json" \
    --output "$OUT_ROOT/compares/legacy_twowiki50_prefill_compare.json"

run_if_missing \
  "$OUT_ROOT/bootstrap/legacy_twowiki50_ewma_vs_fixed_f1_ci.json" \
  bash -lc 'PYTHONPATH=src .venv/bin/python scripts/agnostic_cpu_rag/paired_bootstrap_ci.py \
    "$0" "$1" --metric-path answer_metrics_per_query.f1 > "$2"' \
  "$LEGACY_TWOWIKI_FIXED_DIR/per_query.jsonl" \
  "$LEGACY_TWOWIKI_EWMA_DIR/per_query.jsonl" \
  "$OUT_ROOT/bootstrap/legacy_twowiki50_ewma_vs_fixed_f1_ci.json"

echo "[done] P3-P6 artifacts ready under $OUT_ROOT"
