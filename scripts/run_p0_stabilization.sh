#!/usr/bin/env bash
set -euo pipefail

# P0 stabilization batch launcher (S1-S4)
#
# Goal:
# - Run four long-run configurations after Batch D failure.
# - Keep all shared baseline knobs fixed, only vary (aps, interval).
#
# Usage:
#   ./scripts/run_p0_stabilization.sh
#   J_MAX=2 RUN_TAG=p0_lab1 ./scripts/run_p0_stabilization.sh
#
# Note:
# - Uses only project venv: ./venv/bin/python
# - Writes manifest + logs for reproducibility.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found: $PYTHON_BIN" >&2
  exit 1
fi

J_MAX="${J_MAX:-2}"
RUN_TAG="${RUN_TAG:-p0_stable_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/p0_stabilization/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-logs/$RUN_TAG}"

EVENTS_JSON="docs/personality_dungeon_v1/02_event_templates_v1.json"

mkdir -p "$OUT_DIR" "$LOG_DIR"

COMMON_FLAGS=(
  --enable-events --events-json "$EVENTS_JSON"
  --popularity-mode sampled
  --seed 45 --rounds 12000 --players 300
  --payoff-mode matrix_ab --a 0.8 --b 0.9
  --matrix-cross-coupling 0.16
  --selection-strength 0.06 --init-bias 0.12
  --event-failure-threshold 0.72 --event-health-penalty 0.10
  --adaptive-payoff-target 0.27
)

# id:aps:interval:outfile
CONFIGS=(
  "S1:2.0:400:stable_s2p0_i400_c0p16_seed45_r12000.csv"
  "S2:3.0:400:stable_s3p0_i400_c0p16_seed45_r12000.csv"
  "S3:1.5:800:stable_s1p5_i800_c0p16_seed45_r12000.csv"
  "S4:2.0:800:stable_s2p0_i800_c0p16_seed45_r12000.csv"
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

launch_sim() {
  local label="$1" out="$2" aps="$3" interval="$4"
  wait_for_slot
  local logfile="$LOG_DIR/_$(basename "$out" .csv).log"
  nohup "$PYTHON_BIN" -m simulation.run_simulation \
    "${COMMON_FLAGS[@]}" \
    --adaptive-payoff-strength "$aps" \
    --payoff-update-interval "$interval" \
    --out "$out" > "$logfile" 2>&1 &
  local pid=$!
  ACTIVE_PIDS+=("$pid")
  PID_LABEL[$pid]="$label"
  PID_LOG[$pid]="$logfile"
  printf "  [LAUNCH] %s\n" "$label"
}

echo "ROOT_DIR   = $ROOT_DIR"
echo "PYTHON_BIN = $PYTHON_BIN"
echo "J_MAX      = $J_MAX"
echo "OUT_DIR    = $OUT_DIR"
echo "LOG_DIR    = $LOG_DIR"
echo "---------------------------------------------"

printf "id\taps\tinterval\trounds\tout\n" > "$OUT_DIR/manifest.tsv"

echo "=== Launch P0 stabilization jobs (S1-S4) ==="
for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r id aps interval fname <<< "$cfg"
  out="$OUT_DIR/$fname"
  printf "%s\t%s\t%s\t12000\t%s\n" "$id" "$aps" "$interval" "$out" >> "$OUT_DIR/manifest.tsv"
  launch_sim "$id aps=$aps interval=$interval" "$out" "$aps" "$interval"
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
echo "CSV     : $OUT_DIR"
echo "LOG     : $LOG_DIR"
echo "MANIFEST: $OUT_DIR/manifest.tsv"
echo "Next    : ./scripts/diag_p0_stabilization.sh $OUT_DIR"
