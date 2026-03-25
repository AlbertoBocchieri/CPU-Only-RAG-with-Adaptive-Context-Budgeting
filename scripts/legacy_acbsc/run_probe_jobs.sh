#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=".venv/bin/python"
BENCH_SCRIPT="scripts/benchmark_suite.py"
COMPARE_SCRIPT="scripts/legacy_acbsc/compare_probe_runs.py"
CONTROLLER_COMPARE_SCRIPT="scripts/legacy_acbsc/compare_controller_eval.py"
LARGE_COMPARE_SCRIPT="scripts/legacy_acbsc/compare_large_eval.py"
BUILD_FULL_ASSETS_SCRIPT="scripts/legacy_acbsc/build_full_eval_assets.py"
OUTPUT_ROOT="results/legacy_acbsc_probe"
SMOKE_ROOT="results/legacy_acbsc_probe_smoke"
REP300_ROOT="results/legacy_acbsc_rep300"
CONTROLLER_EVAL_ROOT="results/legacy_acbsc_controller_eval"
CONTROLLER_SMOKE_ROOT="results/legacy_acbsc_controller_eval_smoke"
FULL_HOTPOT_ROOT="results/legacy_acbsc_full_hotpot"
LARGE_TWOWIKI_ROOT="results/legacy_acbsc_large_twowiki"
HOTPOT_QIDS="qids/hotpot_legacy_acbsc_probe100.json"
TWOWIKI_QIDS="qids/twowiki_legacy_acbsc_probe100.json"
HOTPOT_REP300_QIDS="qids/hotpot_legacy_acbsc_rep300.json"
TWOWIKI_REP300_QIDS="qids/twowiki_legacy_acbsc_rep300.json"
HOTPOT_CANONICAL_REP300_QIDS="results/task_family_weight_search/pools/hotpot_qa/representative_300_qids.json"
TWOWIKI_CANONICAL_HOLDOUT1000_QIDS="results/task_family_weight_search/pools/two_wiki_multihop/holdout_1000_qids.json"
HOTPOT_CANONICAL_FULL7405_QIDS="results/task_family_weight_search/pools/hotpot_qa/full_7405_qids.json"
TWOWIKI_CANONICAL_LARGE6288_QIDS="results/task_family_weight_search/pools/two_wiki_multihop/large_holdout_6288_qids.json"
HOTPOT_Q1_EXISTING_SUMMARY="results/hotpot_q1_sc_pool300/cfg_94299ab2d54e/hotpot_qa/summary.json"
HOTPOT_Q1_EXISTING_PER_QUERY="results/hotpot_q1_sc_pool300/cfg_94299ab2d54e/hotpot_qa/per_query.jsonl"

run_hotpot() {
  local out_root="$1"
  local run_id="$2"
  local n_queries="$3"
  local qids_path="$4"
  PYTHONPATH=src caffeinate -di "$PYTHON_BIN" "$BENCH_SCRIPT" \
    --config configs/legacy_acbsc_probe/hotpot_noacb_p4.yaml \
    --config-2 configs/legacy_acbsc_probe/hotpot_incremental_sc_p4.yaml \
    --config-3 configs/legacy_acbsc_probe/hotpot_agnostic_acb_sc_p4.yaml \
    --dataset hotpot_qa \
    --num-queries "$n_queries" \
    --query-ids-path "$qids_path" \
    --seed 42 \
    --run-id "$run_id" \
    --output-dir "$out_root" \
    --ui-update-every 1
}

run_hotpot_acbsc_only() {
  local out_root="$1"
  local run_id="$2"
  local n_queries="$3"
  local qids_path="$4"
  PYTHONPATH=src caffeinate -di "$PYTHON_BIN" "$BENCH_SCRIPT" \
    --config configs/legacy_acbsc_probe/hotpot_agnostic_acb_sc_p4.yaml \
    --dataset hotpot_qa \
    --num-queries "$n_queries" \
    --query-ids-path "$qids_path" \
    --seed 42 \
    --run-id "$run_id" \
    --output-dir "$out_root" \
    --ui-update-every 1
}

run_twowiki() {
  local out_root="$1"
  local run_id="$2"
  local n_queries="$3"
  local qids_path="$4"
  PYTHONPATH=src caffeinate -di "$PYTHON_BIN" "$BENCH_SCRIPT" \
    --config configs/legacy_acbsc_probe/twowiki_noacb_p4.yaml \
    --config-2 configs/legacy_acbsc_probe/twowiki_incremental_sc_p4.yaml \
    --config-3 configs/legacy_acbsc_probe/twowiki_agnostic_acb_sc_p4.yaml \
    --dataset two_wiki_multihop \
    --num-queries "$n_queries" \
    --query-ids-path "$qids_path" \
    --seed 42 \
    --run-id "$run_id" \
    --output-dir "$out_root" \
    --ui-update-every 1
}

run_twowiki_q1_vs_acbsc() {
  local out_root="$1"
  local run_id="$2"
  local n_queries="$3"
  local qids_path="$4"
  PYTHONPATH=src caffeinate -di "$PYTHON_BIN" "$BENCH_SCRIPT" \
    --config configs/legacy_acbsc_probe/twowiki_incremental_sc_p4.yaml \
    --config-2 configs/legacy_acbsc_probe/twowiki_agnostic_acb_sc_p4.yaml \
    --dataset two_wiki_multihop \
    --num-queries "$n_queries" \
    --query-ids-path "$qids_path" \
    --seed 42 \
    --run-id "$run_id" \
    --output-dir "$out_root" \
    --ui-update-every 1
}

compare_runs() {
  local hotpot_root="$1"
  local twowiki_root="$2"
  local out_dir="$3"
  local prefix="$4"
  PYTHONPATH=src "$PYTHON_BIN" "$COMPARE_SCRIPT" \
    --hotpot-run-root "$hotpot_root" \
    --twowiki-run-root "$twowiki_root" \
    --output-dir "$out_dir" \
    --artifact-prefix "$prefix"
}

compare_controller_eval() {
  PYTHONPATH=src "$PYTHON_BIN" "$CONTROLLER_COMPARE_SCRIPT" \
    --hotpot-baseline-summary "$HOTPOT_Q1_EXISTING_SUMMARY" \
    --hotpot-baseline-per-query "$HOTPOT_Q1_EXISTING_PER_QUERY" \
    --hotpot-candidate-run-root "$CONTROLLER_EVAL_ROOT/legacy_acbsc_hotpot_rep300" \
    --twowiki-run-root "$CONTROLLER_EVAL_ROOT/legacy_acbsc_twowiki_holdout1000" \
    --output-dir "$CONTROLLER_EVAL_ROOT/compare"
}

prepare_full_eval_assets() {
  PYTHONPATH=src "$PYTHON_BIN" "$BUILD_FULL_ASSETS_SCRIPT"
}

run_hotpot_full_acbsc() {
  PYTHONPATH=src caffeinate -di "$PYTHON_BIN" "$BENCH_SCRIPT" \
    --config configs/legacy_acbsc_eval/hotpot_agnostic_acb_sc_full7405_p4.yaml \
    --dataset hotpot_qa \
    --num-queries 7405 \
    --query-ids-path "$HOTPOT_CANONICAL_FULL7405_QIDS" \
    --seed 42 \
    --run-id legacy_acbsc_hotpot_full7405 \
    --output-dir "$FULL_HOTPOT_ROOT" \
    --ui-update-every 1
}

run_twowiki_large_q0() {
  PYTHONPATH=src caffeinate -di "$PYTHON_BIN" "$BENCH_SCRIPT" \
    --config configs/legacy_acbsc_eval/twowiki_q0_large6288_p4.yaml \
    --dataset two_wiki_multihop \
    --num-queries 6288 \
    --query-ids-path "$TWOWIKI_CANONICAL_LARGE6288_QIDS" \
    --seed 42 \
    --run-id legacy_q0_twowiki_large6288 \
    --output-dir "$LARGE_TWOWIKI_ROOT" \
    --ui-update-every 1
}

run_twowiki_large_acbsc() {
  PYTHONPATH=src caffeinate -di "$PYTHON_BIN" "$BENCH_SCRIPT" \
    --config configs/legacy_acbsc_eval/twowiki_agnostic_acb_sc_large6288_p4.yaml \
    --dataset two_wiki_multihop \
    --num-queries 6288 \
    --query-ids-path "$TWOWIKI_CANONICAL_LARGE6288_QIDS" \
    --seed 42 \
    --run-id legacy_acbsc_twowiki_large6288 \
    --output-dir "$LARGE_TWOWIKI_ROOT" \
    --ui-update-every 1
}

run_twowiki_large_both() {
  run_twowiki_large_q0
  run_twowiki_large_acbsc
}

compare_large_twowiki() {
  PYTHONPATH=src "$PYTHON_BIN" "$LARGE_COMPARE_SCRIPT" \
    --baseline-run-root "$LARGE_TWOWIKI_ROOT/legacy_q0_twowiki_large6288" \
    --candidate-run-root "$LARGE_TWOWIKI_ROOT/legacy_acbsc_twowiki_large6288" \
    --dataset two_wiki_multihop \
    --baseline-label legacy_q0_query_expansion \
    --candidate-label new_agnostic_acb_sc \
    --output-dir "$LARGE_TWOWIKI_ROOT/compare" \
    --output-name twowiki_legacy_acbsc_vs_q0_large6288_compare.json
}

case "${1:-}" in
  prepare_full_eval_assets)
    prepare_full_eval_assets
    ;;
  smoke_hotpot)
    run_hotpot "$SMOKE_ROOT" legacy_acbsc_probe_hotpot_smoke5 5 "$HOTPOT_QIDS"
    ;;
  smoke_twowiki)
    run_twowiki "$SMOKE_ROOT" legacy_acbsc_probe_twowiki_smoke5 5 "$TWOWIKI_QIDS"
    ;;
  smoke_all)
    run_hotpot "$SMOKE_ROOT" legacy_acbsc_probe_hotpot_smoke5 5 "$HOTPOT_QIDS"
    run_twowiki "$SMOKE_ROOT" legacy_acbsc_probe_twowiki_smoke5 5 "$TWOWIKI_QIDS"
    ;;
  probe_hotpot)
    run_hotpot "$OUTPUT_ROOT" legacy_acbsc_probe_hotpot100 100 "$HOTPOT_QIDS"
    ;;
  probe_twowiki)
    run_twowiki "$OUTPUT_ROOT" legacy_acbsc_probe_twowiki100 100 "$TWOWIKI_QIDS"
    ;;
  probe_all)
    run_hotpot "$OUTPUT_ROOT" legacy_acbsc_probe_hotpot100 100 "$HOTPOT_QIDS"
    run_twowiki "$OUTPUT_ROOT" legacy_acbsc_probe_twowiki100 100 "$TWOWIKI_QIDS"
    ;;
  rep300_hotpot)
    run_hotpot "$REP300_ROOT" legacy_acbsc_rep300_hotpot 300 "$HOTPOT_REP300_QIDS"
    ;;
  rep300_twowiki)
    run_twowiki "$REP300_ROOT" legacy_acbsc_rep300_twowiki 300 "$TWOWIKI_REP300_QIDS"
    ;;
  rep300_all)
    run_hotpot "$REP300_ROOT" legacy_acbsc_rep300_hotpot 300 "$HOTPOT_REP300_QIDS"
    run_twowiki "$REP300_ROOT" legacy_acbsc_rep300_twowiki 300 "$TWOWIKI_REP300_QIDS"
    ;;
  compare)
    compare_runs \
      "$OUTPUT_ROOT/legacy_acbsc_probe_hotpot100" \
      "$OUTPUT_ROOT/legacy_acbsc_probe_twowiki100" \
      "$OUTPUT_ROOT/compare" \
      "legacy_acbsc_probe"
    ;;
  compare_rep300)
    compare_runs \
      "$REP300_ROOT/legacy_acbsc_rep300_hotpot" \
      "$REP300_ROOT/legacy_acbsc_rep300_twowiki" \
      "$REP300_ROOT/compare" \
      "legacy_acbsc_rep300"
    ;;
  controller_smoke_hotpot)
    run_hotpot_acbsc_only "$CONTROLLER_SMOKE_ROOT" legacy_acbsc_hotpot_rep300_smoke5 5 "$HOTPOT_CANONICAL_REP300_QIDS"
    ;;
  controller_smoke_twowiki)
    run_twowiki_q1_vs_acbsc "$CONTROLLER_SMOKE_ROOT" legacy_acbsc_twowiki_holdout1000_smoke5 5 "$TWOWIKI_CANONICAL_HOLDOUT1000_QIDS"
    ;;
  controller_smoke_all)
    run_hotpot_acbsc_only "$CONTROLLER_SMOKE_ROOT" legacy_acbsc_hotpot_rep300_smoke5 5 "$HOTPOT_CANONICAL_REP300_QIDS"
    run_twowiki_q1_vs_acbsc "$CONTROLLER_SMOKE_ROOT" legacy_acbsc_twowiki_holdout1000_smoke5 5 "$TWOWIKI_CANONICAL_HOLDOUT1000_QIDS"
    ;;
  controller_hotpot)
    run_hotpot_acbsc_only "$CONTROLLER_EVAL_ROOT" legacy_acbsc_hotpot_rep300 300 "$HOTPOT_CANONICAL_REP300_QIDS"
    ;;
  controller_twowiki)
    run_twowiki_q1_vs_acbsc "$CONTROLLER_EVAL_ROOT" legacy_acbsc_twowiki_holdout1000 1000 "$TWOWIKI_CANONICAL_HOLDOUT1000_QIDS"
    ;;
  controller_all)
    run_hotpot_acbsc_only "$CONTROLLER_EVAL_ROOT" legacy_acbsc_hotpot_rep300 300 "$HOTPOT_CANONICAL_REP300_QIDS"
    run_twowiki_q1_vs_acbsc "$CONTROLLER_EVAL_ROOT" legacy_acbsc_twowiki_holdout1000 1000 "$TWOWIKI_CANONICAL_HOLDOUT1000_QIDS"
    ;;
  compare_controller)
    compare_controller_eval
    ;;
  full_hotpot_acbsc)
    prepare_full_eval_assets
    run_hotpot_full_acbsc
    ;;
  large_twowiki_q0)
    prepare_full_eval_assets
    run_twowiki_large_q0
    ;;
  large_twowiki_acbsc)
    prepare_full_eval_assets
    run_twowiki_large_acbsc
    ;;
  large_twowiki_both)
    prepare_full_eval_assets
    run_twowiki_large_both
    ;;
  compare_large_twowiki)
    compare_large_twowiki
    ;;
  pack_windows_bundle)
    prepare_full_eval_assets
    ;;
  *)
    echo "usage: $0 {prepare_full_eval_assets|smoke_hotpot|smoke_twowiki|smoke_all|probe_hotpot|probe_twowiki|probe_all|rep300_hotpot|rep300_twowiki|rep300_all|compare|compare_rep300|controller_smoke_hotpot|controller_smoke_twowiki|controller_smoke_all|controller_hotpot|controller_twowiki|controller_all|compare_controller|full_hotpot_acbsc|large_twowiki_q0|large_twowiki_acbsc|large_twowiki_both|compare_large_twowiki|pack_windows_bundle}" >&2
    exit 1
    ;;
esac
