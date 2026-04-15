#!/usr/bin/env bash
set -euo pipefail

# Mechanism search M2: matrix-cross-coupling x representative (a,b) candidates.
#
# Purpose:
# - Stop spending budget on local a/b refinement around the failed B-center plateau.
# - Test whether cross-coupling c is a primary control axis for rotation strength.
# - Hold popularity/update-rule fixed while scanning c across a small set of
#   representative (a,b) candidates away from the B-center region.
#
# Default representative points:
#   (1.00, 0.90)  lower A-center flank
#   (1.04, 0.90)  empirical A-line expected best
#   (1.076,0.90)  lagged near-neutral root vicinity
#
# Default c grid:
#   c in {0.10, 0.16, 0.22}
#
# Default validation scale:
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
RUN_TAG="${RUN_TAG:-mechanism_cross_coupling_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/mechanism_search/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-logs/$RUN_TAG}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

EVENTS_JSON="docs/personality_dungeon_v1/02_event_templates_v1.json"
SEEDS="${SEEDS:-45 47 49 51 53 55}"
ROUNDS="${ROUNDS:-8000}"
AB_PAIRS="${AB_PAIRS:-1.00:0.90 1.04:0.90 1.076:0.90}"
C_GRID="${C_GRID:-0.10 0.16 0.22}"
POPULARITY_MODE="${POPULARITY_MODE:-expected}"
APS="${APS:-1.5}"
INTERVAL="${INTERVAL:-400}"
TARGET="${TARGET:-0.27}"

COMMON_FLAGS=(
  --enable-events --events-json "$EVENTS_JSON"
  --popularity-mode "$POPULARITY_MODE"
  --players 300
  --payoff-mode matrix_ab
  --selection-strength 0.06 --init-bias 0.12
  --event-failure-threshold 0.72 --event-health-penalty 0.10
  --adaptive-payoff-strength "$APS"
  --payoff-update-interval "$INTERVAL"
  --adaptive-payoff-target "$TARGET"
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
  local label="$1" out="$2" seed="$3" aval="$4" bval="$5" cval="$6"
  wait_for_slot
  local logfile="$LOG_DIR/_$(basename "$out" .csv).log"
  nohup "$PYTHON_BIN" -m simulation.run_simulation \
    "${COMMON_FLAGS[@]}" \
    --seed "$seed" \
    --rounds "$ROUNDS" \
    --a "$aval" \
    --b "$bval" \
    --matrix-cross-coupling "$cval" \
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
echo "POPULARITY_MODE = $POPULARITY_MODE"
echo "APS = $APS"
echo "INTERVAL = $INTERVAL"
echo "TARGET = $TARGET"
echo "J_MAX = $J_MAX"
echo "---------------------------------------------"

printf "seed\ta\tb\tc\trounds\tpopularity_mode\taps\tinterval\ttarget\tout\n" > "$OUT_DIR/manifest_cross_coupling.tsv"

read -r -a SEED_ARR <<< "$SEEDS"
read -r -a PAIR_ARR <<< "$AB_PAIRS"
read -r -a C_ARR <<< "$C_GRID"

echo "=== Launch mechanism search: cross-coupling x representative (a,b) ==="
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
    for seed in "${SEED_ARR[@]}"; do
      out="$OUT_DIR/mech_c${ctag}_a${atag}_b${btag}_seed${seed}_r${ROUNDS}.csv"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$seed" "$aval" "$bval" "$cval" "$ROUNDS" "$POPULARITY_MODE" "$APS" "$INTERVAL" "$TARGET" "$out" \
        >> "$OUT_DIR/manifest_cross_coupling.tsv"
      launch_sim "c=$cval a=$aval b=$bval seed=$seed" "$out" "$seed" "$aval" "$bval" "$cval"
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
echo "MANIFEST: $OUT_DIR/manifest_cross_coupling.tsv"