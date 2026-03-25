#!/bin/zsh
set -euo pipefail

ROOT_DIR=/Users/albertobocchieri/Desktop/rag_from_scratch
cd "$ROOT_DIR"

STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
  echo "usage: $0 {derive|pilot75|confirm300|holdout|final}" >&2
  exit 1
fi
if [[ "$STAGE" == "representative" ]]; then
  echo "stage 'representative' is deprecated; using 'pilot75'" >&2
  STAGE="pilot75"
fi

PYTHONPATH=src
BASE_CFG=configs/agnostic_cpu_rag/fixed_alpha_nohop2_task_family_defaults.yaml
INHERITED_CFG=configs/agnostic_cpu_rag/fixed_alpha_nohop2_multihop_stable.yaml
NO_CONTROLLER_CFG=configs/agnostic_cpu_rag/fixed_alpha_nohop2_no_controller.yaml
MODEL_REGISTRY=configs/agnostic_cpu_rag/model_registry.yaml
ROOT_RESULTS=results/task_family_weight_search
POOLS=$ROOT_RESULTS/pools
CACHES=$ROOT_RESULTS/caches
DERIVED=$ROOT_RESULTS/derived_weights
WINNERS=$ROOT_RESULTS/selection_winners
VALIDATION=$ROOT_RESULTS/validation
FINAL=$ROOT_RESULTS/final_runs
HOTPOT_FULL_QIDS=results/full_7405_best_quality_base_power/cfg_a000fd2bb017/hotpot_qa/sampled_qids.json

run_bench() {
  PYTHONPATH=$PYTHONPATH caffeinate -di .venv/bin/python scripts/agnostic_cpu_rag/run_benchmark.py "$@"
}

run_cache() {
  PYTHONPATH=$PYTHONPATH caffeinate -di .venv/bin/python scripts/agnostic_cpu_rag/export_controller_cache.py "$@"
}

json_get() {
  local path="$1"
  local key_path="$2"
  PYTHONPATH=$PYTHONPATH .venv/bin/python - "$path" "$key_path" <<'PY'
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

candidate_paths_from_shortlist() {
  local shortlist_json="$1"
  local limit="$2"
  PYTHONPATH=$PYTHONPATH .venv/bin/python - "$shortlist_json" "$limit" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
limit = int(sys.argv[2])
if not path.exists():
    raise SystemExit(0)
rows = json.loads(path.read_text(encoding="utf-8"))
seen = set()
count = 0
for row in rows:
    key = json.dumps(row.get("metrics", {}), sort_keys=True)
    if key in seen:
        continue
    seen.add(key)
    print(row["override_path"])
    count += 1
    if count >= limit:
        break
PY
}

winner_report_path() {
  local family="$1"
  echo "$WINNERS/${family}_winner_report.json"
}

winner_exists() {
  local family="$1"
  [[ -f "$(winner_report_path "$family")" ]]
}

winner_field() {
  local family="$1"
  local key_path="$2"
  json_get "$(winner_report_path "$family")" "$key_path"
}

case "$STAGE" in
  derive)
    rm -rf "$DERIVED" "$WINNERS"
    mkdir -p "$DERIVED" "$WINNERS"

    PYTHONPATH=$PYTHONPATH .venv/bin/python scripts/agnostic_cpu_rag/build_task_family_pools.py \
      --output-root "$POOLS"

    run_cache \
      --config "$BASE_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset hotpot_qa \
      --include-qids-path "$POOLS/hotpot_qa/tuning_150_qids.json" \
      --run-id hotpot_tuning_150_cache \
      --output-dir "$CACHES" \
      --pool-role tuning

    run_cache \
      --config "$BASE_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset two_wiki_multihop \
      --include-qids-path "$POOLS/two_wiki_multihop/tuning_150_qids.json" \
      --run-id twowiki_tuning_150_cache \
      --output-dir "$CACHES" \
      --pool-role tuning

    run_cache \
      --config "$BASE_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset squad_open \
      --include-qids-path "$POOLS/squad_open/tuning_150_qids.json" \
      --run-id squad_tuning_150_cache \
      --output-dir "$CACHES" \
      --pool-role tuning

    PYTHONPATH=$PYTHONPATH .venv/bin/python scripts/agnostic_cpu_rag/derive_task_family_weights.py multi_hop \
      --hotpot-cache "$CACHES/hotpot_tuning_150_cache/hotpot_qa/controller_cache.jsonl" \
      --twowiki-cache "$CACHES/twowiki_tuning_150_cache/two_wiki_multihop/controller_cache.jsonl" \
      --controller-cfg-yaml "$BASE_CFG" \
      --output-dir "$DERIVED"

    PYTHONPATH=$PYTHONPATH .venv/bin/python scripts/agnostic_cpu_rag/derive_task_family_weights.py open_qa \
      --squad-cache "$CACHES/squad_tuning_150_cache/squad_open/controller_cache.jsonl" \
      --controller-cfg-yaml "$BASE_CFG" \
      --output-dir "$DERIVED"
    ;;

  pilot75)
    mkdir -p "$VALIDATION" "$WINNERS"

    run_bench \
      --config "$INHERITED_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset hotpot_qa \
      --include-qids-path "$POOLS/hotpot_qa/selection_pilot_75_qids.json" \
      --run-id hotpot_pilot75_inherited \
      --output-dir "$VALIDATION" \
      --pool-role selection_pilot \
      --weights-source inherited_legacy

    run_bench \
      --config "$INHERITED_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset two_wiki_multihop \
      --include-qids-path "$POOLS/two_wiki_multihop/selection_pilot_75_qids.json" \
      --run-id twowiki_pilot75_inherited \
      --output-dir "$VALIDATION" \
      --pool-role selection_pilot \
      --weights-source inherited_legacy

    for CANDIDATE in ${(f)"$(candidate_paths_from_shortlist "$DERIVED/multi_hop_qa_shortlist.json" 3)"}; do
      [[ -f "$CANDIDATE" ]] || continue
      CNAME=$(basename "$CANDIDATE" .yaml)
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$CANDIDATE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset hotpot_qa \
        --include-qids-path "$POOLS/hotpot_qa/selection_pilot_75_qids.json" \
        --run-id "hotpot_pilot75_${CNAME}" \
        --output-dir "$VALIDATION" \
        --pool-role selection_pilot \
        --weights-source task_family_candidate
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$CANDIDATE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset two_wiki_multihop \
        --include-qids-path "$POOLS/two_wiki_multihop/selection_pilot_75_qids.json" \
        --run-id "twowiki_pilot75_${CNAME}" \
        --output-dir "$VALIDATION" \
        --pool-role selection_pilot \
        --weights-source task_family_candidate
    done

    run_bench \
      --config "$INHERITED_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset squad_open \
      --include-qids-path "$POOLS/squad_open/selection_pilot_75_qids.json" \
      --run-id squad_pilot75_inherited \
      --output-dir "$VALIDATION" \
      --pool-role selection_pilot \
      --weights-source inherited_legacy

    for CANDIDATE in ${(f)"$(candidate_paths_from_shortlist "$DERIVED/open_qa_shortlist.json" 2)"}; do
      [[ -f "$CANDIDATE" ]] || continue
      CNAME=$(basename "$CANDIDATE" .yaml)
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$CANDIDATE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset squad_open \
        --include-qids-path "$POOLS/squad_open/selection_pilot_75_qids.json" \
        --run-id "squad_pilot75_${CNAME}" \
        --output-dir "$VALIDATION" \
        --pool-role selection_pilot \
        --weights-source task_family_candidate
    done

    run_bench \
      --config "$NO_CONTROLLER_CFG" \
      --model-registry "$MODEL_REGISTRY" \
      --dataset squad_open \
      --include-qids-path "$POOLS/squad_open/selection_pilot_75_qids.json" \
      --run-id squad_pilot75_no_controller \
      --output-dir "$VALIDATION" \
      --pool-role selection_pilot \
      --weights-source no_controller_baseline

    PYTHONPATH=$PYTHONPATH .venv/bin/python scripts/agnostic_cpu_rag/select_task_family_weight_winners.py pilot75 \
      --derived-dir "$DERIVED" \
      --validation-dir "$VALIDATION" \
      --output-dir "$WINNERS"
    ;;

  confirm300)
    [[ -f "$WINNERS/multi_hop_winner_report.json" ]] || { echo "run pilot75 first" >&2; exit 1; }
    [[ -f "$WINNERS/open_qa_winner_report.json" ]] || { echo "run pilot75 first" >&2; exit 1; }

    if [[ "$(winner_field multi_hop winner_type)" == "candidate" ]]; then
      MULTI_HOP_OVERRIDE="$(winner_field multi_hop winner_override_path)"
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$MULTI_HOP_OVERRIDE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset hotpot_qa \
        --include-qids-path "$POOLS/hotpot_qa/representative_300_qids.json" \
        --run-id hotpot_rep300_winner \
        --output-dir "$VALIDATION" \
        --pool-role representative_validation \
        --weights-source task_family_winner
      run_bench \
        --config "$INHERITED_CFG" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset hotpot_qa \
        --include-qids-path "$POOLS/hotpot_qa/representative_300_qids.json" \
        --run-id hotpot_rep300_inherited \
        --output-dir "$VALIDATION" \
        --pool-role representative_validation \
        --weights-source inherited_legacy
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$MULTI_HOP_OVERRIDE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset two_wiki_multihop \
        --include-qids-path "$POOLS/two_wiki_multihop/representative_300_qids.json" \
        --run-id twowiki_rep300_winner \
        --output-dir "$VALIDATION" \
        --pool-role representative_validation \
        --weights-source task_family_winner
      run_bench \
        --config "$INHERITED_CFG" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset two_wiki_multihop \
        --include-qids-path "$POOLS/two_wiki_multihop/representative_300_qids.json" \
        --run-id twowiki_rep300_inherited \
        --output-dir "$VALIDATION" \
        --pool-role representative_validation \
        --weights-source inherited_legacy
    else
      echo "multi_hop winner retained inherited; skipping representative300 multi-hop candidate runs" >&2
    fi

    if [[ "$(winner_field open_qa winner_type)" == "candidate" ]]; then
      OPEN_QA_OVERRIDE="$(winner_field open_qa winner_override_path)"
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$OPEN_QA_OVERRIDE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset squad_open \
        --include-qids-path "$POOLS/squad_open/representative_300_qids.json" \
        --run-id squad_rep300_winner \
        --output-dir "$VALIDATION" \
        --pool-role representative_validation \
        --weights-source task_family_winner
      run_bench \
        --config "$INHERITED_CFG" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset squad_open \
        --include-qids-path "$POOLS/squad_open/representative_300_qids.json" \
        --run-id squad_rep300_inherited \
        --output-dir "$VALIDATION" \
        --pool-role representative_validation \
        --weights-source inherited_legacy
      run_bench \
        --config "$NO_CONTROLLER_CFG" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset squad_open \
        --include-qids-path "$POOLS/squad_open/representative_300_qids.json" \
        --run-id squad_rep300_no_controller \
        --output-dir "$VALIDATION" \
        --pool-role representative_validation \
        --weights-source no_controller_baseline
    else
      echo "open_qa winner retained inherited; skipping representative300 open_qa candidate runs" >&2
    fi

    PYTHONPATH=$PYTHONPATH .venv/bin/python scripts/agnostic_cpu_rag/select_task_family_weight_winners.py confirm300 \
      --derived-dir "$DERIVED" \
      --validation-dir "$VALIDATION" \
      --output-dir "$WINNERS"
    ;;

  holdout)
    [[ -f "$WINNERS/multi_hop_winner_report.json" ]] || { echo "run pilot75/confirm300 first" >&2; exit 1; }
    [[ -f "$WINNERS/open_qa_winner_report.json" ]] || { echo "run pilot75/confirm300 first" >&2; exit 1; }

    if [[ "$(winner_field multi_hop representative300_status)" == "passed" ]]; then
      MULTI_HOP_OVERRIDE="$(winner_field multi_hop winner_override_path)"
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
      run_bench \
        --config "$INHERITED_CFG" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset two_wiki_multihop \
        --include-qids-path "$POOLS/two_wiki_multihop/holdout_1000_qids.json" \
        --run-id twowiki_holdout1000_inherited \
        --output-dir "$VALIDATION" \
        --pool-role large_holdout \
        --weights-source inherited_legacy
    else
      echo "multi_hop not confirmed at representative300; skipping holdout" >&2
    fi

    if [[ "$(winner_field open_qa representative300_status)" == "passed" ]]; then
      OPEN_QA_OVERRIDE="$(winner_field open_qa winner_override_path)"
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
    else
      echo "open_qa not confirmed at representative300; skipping holdout" >&2
    fi
    ;;

  final)
    [[ -f "$WINNERS/multi_hop_winner_report.json" ]] || { echo "run pilot75/confirm300 first" >&2; exit 1; }
    [[ -f "$WINNERS/open_qa_winner_report.json" ]] || { echo "run pilot75/confirm300 first" >&2; exit 1; }

    if [[ "$(winner_field multi_hop representative300_status)" == "passed" ]]; then
      MULTI_HOP_OVERRIDE="$(winner_field multi_hop winner_override_path)"
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$MULTI_HOP_OVERRIDE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset hotpot_qa \
        --include-qids-path "$HOTPOT_FULL_QIDS" \
        --run-id hotpot_full7405_winner \
        --output-dir "$FINAL" \
        --pool-role full \
        --weights-source task_family_winner
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$MULTI_HOP_OVERRIDE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset two_wiki_multihop \
        --num-queries 12576 \
        --disable-llm \
        --run-id twowiki_full12576_controller_only_winner \
        --output-dir "$FINAL" \
        --pool-role full \
        --weights-source task_family_winner
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$MULTI_HOP_OVERRIDE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset two_wiki_multihop \
        --include-qids-path "$POOLS/two_wiki_multihop/holdout_1000_qids.json" \
        --run-id twowiki_final1000_winner \
        --output-dir "$FINAL" \
        --pool-role full \
        --weights-source task_family_winner
    else
      echo "multi_hop not confirmed at representative300; skipping final multi-hop runs" >&2
    fi

    if [[ "$(winner_field open_qa representative300_status)" == "passed" ]]; then
      OPEN_QA_OVERRIDE="$(winner_field open_qa winner_override_path)"
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$OPEN_QA_OVERRIDE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset squad_open \
        --num-queries 10570 \
        --disable-llm \
        --run-id squad_full10570_controller_only_winner \
        --output-dir "$FINAL" \
        --pool-role full \
        --weights-source task_family_winner
      run_bench \
        --config "$BASE_CFG" \
        --config-override "$OPEN_QA_OVERRIDE" \
        --model-registry "$MODEL_REGISTRY" \
        --dataset squad_open \
        --include-qids-path "$POOLS/squad_open/holdout_1000_qids.json" \
        --run-id squad_final1000_winner \
        --output-dir "$FINAL" \
        --pool-role full \
        --weights-source task_family_winner
    else
      echo "open_qa not confirmed at representative300; skipping final open_qa runs" >&2
    fi
    ;;

  *)
    echo "unknown stage: $STAGE" >&2
    exit 1
    ;;
esac
