#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SENTINEL="results/final_acb_sc/agnostic_hotpot1000_multihop_stable_ewma/hotpot_qa/summary.json"
LOG="results/final_acb_sc/logs/p3_p6_campaign.log"
mkdir -p "$(dirname "$LOG")"
echo "[$(date '+%F %T')] waiter started" >> "$LOG"
while [[ ! -f "$SENTINEL" ]]; do
  sleep 60
done
echo "[$(date '+%F %T')] sentinel detected" >> "$LOG"
scripts/agnostic_cpu_rag/run_phase_p3_p6.sh >> "$LOG" 2>&1
