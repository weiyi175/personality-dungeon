#!/usr/bin/env bash
set -euo pipefail

# 2D expected grid around A-center best candidate.
#
# Default grid:
#   a in {1.00, 1.04, 1.08}
#   b in {0.86, 0.90, 0.94}
#   popularity-mode = expected
#   seeds = {45,47,49,51,53,55}
#   rounds = 8000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found: $PYTHON_BIN" >&2
  exit 1
fi

J_MAX="${J_MAX:-12}"
RUN_TAG="${RUN_TAG:-hopf_grid2d_expected_a_center_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/hopf_grid2d/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-logs/$RUN_TAG}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

EVENTS_JSON="docs/personality_dungeon_v1/02_event_templates_v1.json"
SEEDS="${SEEDS:-45 47 49 51 53 55}"
ROUNDS="${ROUNDS:-8000}"
A_GRID="${A_GRID:-1.00 1.04 1.08}"
B_GRID="${B_GRID:-0.86 0.90 0.94}"
POPULARITY_MODE="${POPULARITY_MODE:-expected}"

COMMON_FLAGS=(
  --enable-events --events-json "$EVENTS_JSON"
  --popularity-mode "$POPULARITY_MODE"
  --players 300
  --payoff-mode matrix_ab
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

launch_sim() {
  local label="$1" out="$2" seed="$3" aval="$4" bval="$5"
  wait_for_slot
  local logfile="$LOG_DIR/_$(basename "$out" .csv).log"
  nohup "$PYTHON_BIN" -m simulation.run_simulation \
    "${COMMON_FLAGS[@]}" \
    --seed "$seed" \
    --rounds "$ROUNDS" \
    --a "$aval" \
    --b "$bval" \
    --out "$out" > "$logfile" 2>&1 &
  local pid=$!
  ACTIVE_PIDS+=("$pid")
  PID_LABEL[$pid]="$label"
  PID_LOG[$pid]="$logfile"
  printf "  [LAUNCH] %s\n" "$label"
}

echo "OUT_DIR = $OUT_DIR"
echo "LOG_DIR = $LOG_DIR"
echo "SEEDS = $SEEDS"
echo "ROUNDS = $ROUNDS"
echo "A_GRID = $A_GRID"
echo "B_GRID = $B_GRID"
echo "POPULARITY_MODE = $POPULARITY_MODE"
echo "J_MAX = $J_MAX"
echo "---------------------------------------------"

printf "seed\ta\tb\trounds\tout\n" > "$OUT_DIR/manifest.tsv"
read -r -a SEED_ARR <<< "$SEEDS"
read -r -a A_ARR <<< "$A_GRID"
read -r -a B_ARR <<< "$B_GRID"

echo "=== Launch 2D expected grid around A-center ==="
for aval in "${A_ARR[@]}"; do
  atag="$(printf '%s' "$aval" | tr '.' 'p')"
  for bval in "${B_ARR[@]}"; do
    btag="$(printf '%s' "$bval" | tr '.' 'p')"
    for seed in "${SEED_ARR[@]}"; do
      out="$OUT_DIR/hopf_gridA_a${atag}_b${btag}_seed${seed}_r${ROUNDS}.csv"
      printf "%s\t%s\t%s\t%s\t%s\n" "$seed" "$aval" "$bval" "$ROUNDS" "$out" >> "$OUT_DIR/manifest.tsv"
      launch_sim "a=$aval b=$bval seed=$seed" "$out" "$seed" "$aval" "$bval"
    done
  done
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
