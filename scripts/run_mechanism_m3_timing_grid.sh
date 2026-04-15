#!/usr/bin/env bash
set -euo pipefail

# Mechanism search M3: update timing/target x representative (a,b), under the
# best currently supported background mechanism.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found: $PYTHON_BIN" >&2
  exit 1
fi

J_MAX="${J_MAX:-12}"
RUN_TAG="${RUN_TAG:-mechanism_m3_timing_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/mechanism_search/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-logs/$RUN_TAG}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

EVENTS_JSON="docs/personality_dungeon_v1/02_event_templates_v1.json"
SEEDS="${SEEDS:-45 47 49 51 53 55}"
ROUNDS="${ROUNDS:-8000}"
AB_PAIRS="${AB_PAIRS:-1.00:0.90 1.076:0.90}"
C_GRID="${C_GRID:-0.16 0.20}"
INTERVAL_GRID="${INTERVAL_GRID:-200 400 800}"
TARGET_GRID="${TARGET_GRID:-0.25 0.27 0.29}"
POPULARITY_MODE="${POPULARITY_MODE:-sampled}"
APS="${APS:-1.5}"

COMMON_FLAGS=(
  --enable-events --events-json "$EVENTS_JSON"
  --popularity-mode "$POPULARITY_MODE"
  --players 300
  --payoff-mode matrix_ab
  --selection-strength 0.06 --init-bias 0.12
  --event-failure-threshold 0.72 --event-health-penalty 0.10
  --adaptive-payoff-strength "$APS"
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
  local label="$1" out="$2" seed="$3" aval="$4" bval="$5" cval="$6" interval="$7" target="$8"
  wait_for_slot
  local logfile="$LOG_DIR/_$(basename "$out" .csv).log"
  nohup "$PYTHON_BIN" -m simulation.run_simulation \
    "${COMMON_FLAGS[@]}" \
    --seed "$seed" \
    --rounds "$ROUNDS" \
    --a "$aval" \
    --b "$bval" \
    --matrix-cross-coupling "$cval" \
    --payoff-update-interval "$interval" \
    --adaptive-payoff-target "$target" \
    --out "$out" > "$logfile" 2>&1 &
  local pid=$!
  ACTIVE_PIDS+=("$pid")
  PID_LABEL[$pid]="$label"
  PID_LOG[$pid]="$logfile"
  printf "  [LAUNCH] %s\n" "$label"
}

echo "ROOT_DIR = $ROOT_DIR"
echo "PYTHON_BIN = $PYTHON_BIN"
echo "OUT_DIR = $OUT_DIR"
echo "LOG_DIR = $LOG_DIR"
echo "SEEDS = $SEEDS"
echo "ROUNDS = $ROUNDS"
echo "AB_PAIRS = $AB_PAIRS"
echo "C_GRID = $C_GRID"
echo "INTERVAL_GRID = $INTERVAL_GRID"
echo "TARGET_GRID = $TARGET_GRID"
echo "POPULARITY_MODE = $POPULARITY_MODE"
echo "APS = $APS"
echo "J_MAX = $J_MAX"
echo "---------------------------------------------"

printf "seed\ta\tb\tc\trounds\tpopularity_mode\taps\tinterval\ttarget\tout\n" > "$OUT_DIR/manifest_m3_timing.tsv"

read -r -a SEED_ARR <<< "$SEEDS"
read -r -a PAIR_ARR <<< "$AB_PAIRS"
read -r -a C_ARR <<< "$C_GRID"
read -r -a INTERVAL_ARR <<< "$INTERVAL_GRID"
read -r -a TARGET_ARR <<< "$TARGET_GRID"

echo "=== Launch mechanism search: M3 timing x target ==="
for pair in "${PAIR_ARR[@]}"; do
  IFS=':' read -r aval bval <<< "$pair"
  if [[ -z "$aval" || -z "$bval" ]]; then
    echo "ERROR: invalid pair '$pair' (expected a:b)" >&2
    exit 1
  fi
  atag="$(printf '%s' "$aval" | tr '.' 'p')"
  btag="$(printf '%s' "$bval" | tr '.' 'p')"
  for cval in "${C_ARR[@]}"; do
    ctag="$(printf '%s' "$cval" | tr '.' 'p')"
    for interval in "${INTERVAL_ARR[@]}"; do
      for target in "${TARGET_ARR[@]}"; do
        ttag="$(printf '%s' "$target" | tr '.' 'p')"
        for seed in "${SEED_ARR[@]}"; do
          out="$OUT_DIR/mech_m3_c${ctag}_a${atag}_b${btag}_i${interval}_t${ttag}_seed${seed}_r${ROUNDS}.csv"
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$seed" "$aval" "$bval" "$cval" "$ROUNDS" "$POPULARITY_MODE" "$APS" "$interval" "$target" "$out" \
            >> "$OUT_DIR/manifest_m3_timing.tsv"
          launch_sim \
            "c=$cval a=$aval b=$bval interval=$interval target=$target seed=$seed" \
            "$out" "$seed" "$aval" "$bval" "$cval" "$interval" "$target"
        done
      done
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
echo "MANIFEST: $OUT_DIR/manifest_m3_timing.tsv"