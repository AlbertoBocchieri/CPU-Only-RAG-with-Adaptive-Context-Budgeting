#!/bin/zsh
set -euo pipefail

ROOT_DIR=/Users/albertobocchieri/Desktop/rag_from_scratch
cd "$ROOT_DIR"

BATCH="${1:-}"
if [[ -z "$BATCH" ]]; then
  echo "usage: $0 {batch1|batch2}" >&2
  exit 1
fi

PYTHONPATH=src
BASE_CFG=configs/agnostic_cpu_rag/fixed_alpha_nohop2_task_family_defaults.yaml
INHERITED_CFG=configs/agnostic_cpu_rag/fixed_alpha_nohop2_multihop_stable.yaml
NO_CONTROLLER_CFG=configs/agnostic_cpu_rag/fixed_alpha_nohop2_no_controller.yaml
MODEL_REGISTRY=configs/agnostic_cpu_rag/model_registry.yaml
ROOT_RESULTS=results/task_family_weight_search
POOLS=$ROOT_RESULTS/pools
WINNERS=$ROOT_RESULTS/selection_winners
VALIDATION=$ROOT_RESULTS/validation

winner_field() {
  local family="$1"
  local key_path="$2"
  PYTHONPATH=$PYTHONPATH .venv/bin/python - "$WINNERS/${family}_winner_report.json" "$key_path" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
key_path = sys.argv[2]
if not path.exists():
    raise SystemExit(1)
node = json.loads(path.read_text(encoding="utf-8"))
for part in key_path.split('.'):
    if isinstance(node, dict):
        node = node.get(part)
    else:
        node = None
        break
if node is None:
    raise SystemExit(1)
if isinstance(node, (dict, list)):
    print(json.dumps(node))
else:
    print(node)
PY
}

run_bench() {
  local run_id=""
  local dataset=""
  local output_dir=""
  local args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-id)
        run_id="$2"
        args+=("$1" "$2")
        shift 2
        ;;
      --dataset)
        dataset="$2"
        args+=("$1" "$2")
        shift 2
        ;;
      --output-dir)
        output_dir="$2"
        args+=("$1" "$2")
        shift 2
        ;;
      *)
        args+=("$1")
        shift
        ;;
    esac
  done

  if [[ -z "$run_id" || -z "$dataset" || -z "$output_dir" ]]; then
    echo "run_bench missing required metadata" >&2
    exit 1
  fi

  local summary_path="$output_dir/$run_id/$dataset/summary.json"
  if [[ -f "$summary_path" ]]; then
    echo "skip existing: $summary_path" >&2
    return 0
  fi

  PYTHONPATH=$PYTHONPATH caffeinate -di .venv/bin/python scripts/agnostic_cpu_rag/run_benchmark.py "${args[@]}"
}

[[ -f "$WINNERS/multi_hop_winner_report.json" ]] || { echo "missing multi_hop winner report; run confirm300 first" >&2; exit 1; }
[[ -f "$WINNERS/open_qa_winner_report.json" ]] || { echo "missing open_qa winner report; run confirm300 first" >&2; exit 1; }

[[ "$(winner_field multi_hop representative300_status)" == "passed" ]] || {
  echo "multi_hop representative300_status is not passed; refusing holdout batch" >&2
  exit 1
}
[[ "$(winner_field open_qa representative300_status)" == "passed" ]] || {
  echo "open_qa representative300_status is not passed; refusing holdout batch" >&2
  exit 1
}

MULTI_HOP_OVERRIDE="$(winner_field multi_hop winner_override_path)"
OPEN_QA_OVERRIDE="$(winner_field open_qa winner_override_path)"

case "$BATCH" in
  batch1)
    run_bench \
      --config "$BASE_CFG" \
      --config-override "$MULTI_HOP_OVERRIDE" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset hotpot_qa \
      --include-qids-path "$POOLS/hotpot_qa/holdout_1000_qids.json" \
      --run-id hotpot_holdout1000_winner \
      --output-dir "$VALIDATION" \
      --pool-role large_holdout \
      --weights-source task_family_winner

    run_bench \
      --config "$INHERITED_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset hotpot_qa \
      --include-qids-path "$POOLS/hotpot_qa/holdout_1000_qids.json" \
      --run-id hotpot_holdout1000_inherited \
      --output-dir "$VALIDATION" \
      --pool-role large_holdout \
      --weights-source inherited_legacy

    run_bench \
      --config "$BASE_CFG" \
      --config-override "$MULTI_HOP_OVERRIDE" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset two_wiki_multihop \
      --include-qids-path "$POOLS/two_wiki_multihop/holdout_1000_qids.json" \
      --run-id twowiki_holdout1000_winner \
      --output-dir "$VALIDATION" \
      --pool-role large_holdout \
      --weights-source task_family_winner
    ;;

  batch2)
    run_bench \
      --config "$INHERITED_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset two_wiki_multihop \
      --include-qids-path "$POOLS/two_wiki_multihop/holdout_1000_qids.json" \
      --run-id twowiki_holdout1000_inherited \
      --output-dir "$VALIDATION" \
      --pool-role large_holdout \
      --weights-source inherited_legacy

    run_bench \
      --config "$BASE_CFG" \
      --config-override "$OPEN_QA_OVERRIDE" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset squad_open \
      --include-qids-path "$POOLS/squad_open/holdout_1000_qids.json" \
      --run-id squad_holdout1000_winner \
      --output-dir "$VALIDATION" \
      --pool-role large_holdout \
      --weights-source task_family_winner

    run_bench \
      --config "$INHERITED_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset squad_open \
      --include-qids-path "$POOLS/squad_open/holdout_1000_qids.json" \
      --run-id squad_holdout1000_inherited \
      --output-dir "$VALIDATION" \
      --pool-role large_holdout \
      --weights-source inherited_legacy

    run_bench \
      --config "$NO_CONTROLLER_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset squad_open \
      --include-qids-path "$POOLS/squad_open/holdout_1000_qids.json" \
      --run-id squad_holdout1000_no_controller \
      --output-dir "$VALIDATION" \
      --pool-role large_holdout \
      --weights-source no_controller_baseline
    ;;

  *)
    echo "unknown batch: $BATCH" >&2
    exit 1
    ;;
esac
