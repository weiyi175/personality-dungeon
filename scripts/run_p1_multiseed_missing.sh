#!/usr/bin/env bash
set -euo pipefail

# Run missing seeds for Batch B full set.
# Existing light seeds were 45/48/51; runbook target is 45/47/49/51/53/55.
# This script fills the missing four: 47, 49, 53, 55.
#
# Usage:
#   ./scripts/run_p1_multiseed_missing.sh
#   J_MAX=4 ./scripts/run_p1_multiseed_missing.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found: $PYTHON_BIN" >&2
  exit 1
fi

J_MAX="${J_MAX:-4}"
RUN_TAG="${RUN_TAG:-p1_multiseed_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/$RUN_TAG}"
mkdir -p "$LOG_DIR"

EVENTS_JSON="docs/personality_dungeon_v1/02_event_templates_v1.json"
SEEDS=(47 49 53 55)

COMMON_FLAGS=(
  --enable-events --events-json "$EVENTS_JSON"
  --popularity-mode sampled
  --rounds 8000 --players 300
  --payoff-mode matrix_ab --a 0.8 --b 0.9
  --matrix-cross-coupling 0.16
  --selection-strength 0.06 --init-bias 0.12
  --event-failure-threshold 0.72 --event-health-penalty 0.10
  --adaptive-payoff-strength 1.5
  --payoff-update-interval 400
  --adaptive-payoff-target 0.27
)

ACTIVE_PIDS=()
declare -A PID_LABEL PID_LOG
FAILED=0

reap_finished() {
  local new_active=()
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_active+=("$pid")
      continue
    fi
    if wait "$pid"; then
      printf "  [DONE]   %s\n" "${PID_LABEL[$pid]}"
    else
      echo "WARN: failed: ${PID_LABEL[$pid]}" >&2
      echo "      log=${PID_LOG[$pid]}" >&2
      FAILED=$((FAILED + 1))
    fi
  done
  ACTIVE_PIDS=("${new_active[@]}")
}

wait_for_slot() {
  while [[ ${#ACTIVE_PIDS[@]} -ge $J_MAX ]]; do
    sleep 1
    reap_finished
  done
}

wait_for_all() {
  while [[ ${#ACTIVE_PIDS[@]} -gt 0 ]]; do
    sleep 1
    reap_finished
  done
}

launch_seed() {
  local seed="$1"
  wait_for_slot
  local out="outputs/multiseed_crossc0p16_seed${seed}.csv"
  local log="$LOG_DIR/_multiseed_crossc0p16_seed${seed}.log"
  nohup "$PYTHON_BIN" -m simulation.run_simulation \
    "${COMMON_FLAGS[@]}" \
    --seed "$seed" \
    --out "$out" > "$log" 2>&1 &
  local pid=$!
  ACTIVE_PIDS+=("$pid")
  PID_LABEL[$pid]="seed=$seed"
  PID_LOG[$pid]="$log"
  printf "  [LAUNCH] seed=%s out=%s\n" "$seed" "$out"
}

echo "ROOT_DIR   = $ROOT_DIR"
echo "PYTHON_BIN = $PYTHON_BIN"
echo "J_MAX      = $J_MAX"
echo "LOG_DIR    = $LOG_DIR"
echo "---------------------------------------------"

echo "=== Launch missing Batch-B seeds ==="
for seed in "${SEEDS[@]}"; do
  launch_seed "$seed"
done

echo
echo "=== Waiting for simulations ==="
wait_for_all

if [[ $FAILED -gt 0 ]]; then
  echo "WARN: $FAILED jobs failed" >&2
  exit 2
fi

echo
echo "DONE"
echo "Generated CSVs:"
for seed in "${SEEDS[@]}"; do
  echo "  outputs/multiseed_crossc0p16_seed${seed}.csv"
done

echo "Next diagnosis example:"
echo "  ./venv/bin/python scripts/diagnose_cycle.py outputs/multiseed_crossc0p16_seed47.csv outputs/multiseed_crossc0p16_seed49.csv outputs/multiseed_crossc0p16_seed53.csv outputs/multiseed_crossc0p16_seed55.csv"
