#!/usr/bin/env bash
set -euo pipefail

# Structural search for near-Hopf / near-neutral (a, b) candidates.
#
# Goal:
# - Stop tuning within the failed c=0.16 basin.
# - Scan local linearization around the current matrix_ab baseline.
# - Find candidate (a, b) where:
#     lagged mode: rho - 1 ~= 0
#     ode mode:    alpha ~= 0
#
# Usage:
#   ./scripts/run_hopf_structural_search.sh
#   RUN_TAG=hopf_local_1 ./scripts/run_hopf_structural_search.sh
#   A0=0.8 B0=0.9 A_MIN=0.4 A_MAX=1.2 B_MIN=0.4 B_MAX=1.2 N=81 ./scripts/run_hopf_structural_search.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found: $PYTHON_BIN" >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-hopf_search_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/hopf_scan/$RUN_TAG}"
mkdir -p "$OUT_DIR"

A0="${A0:-0.8}"
B0="${B0:-0.9}"
K="${K:-0.06}"
N="${N:-81}"
FD_H="${FD_H:-1e-6}"
TOL="${TOL:-1e-6}"
REFINE_ITERS="${REFINE_ITERS:-60}"

A_MIN="${A_MIN:-0.4}"
A_MAX="${A_MAX:-1.2}"
B_MIN="${B_MIN:-0.4}"
B_MAX="${B_MAX:-1.2}"

echo "ROOT_DIR = $ROOT_DIR"
echo "PYTHON_BIN = $PYTHON_BIN"
echo "OUT_DIR = $OUT_DIR"
echo "A0 = $A0"
echo "B0 = $B0"
echo "K = $K"
echo "A range = [$A_MIN, $A_MAX]"
echo "B range = [$B_MIN, $B_MAX]"
echo "N = $N"
echo "---------------------------------------------"

run_scan() {
  local mode="$1"
  local axis="$2"
  local outfile="$3"
  shift 3
  echo "=== $mode / scan $axis -> $outfile ==="
  "$PYTHON_BIN" -m analysis.hopf_scan "$@" | tee "$outfile"
  echo
}

run_scan \
  lagged b "$OUT_DIR/lagged_scan_b.txt" \
  --mode lagged --scan b \
  --a "$A0" --b "$B0" \
  --b-min "$B_MIN" --b-max "$B_MAX" --n "$N" \
  --selection-strength "$K" --fd-h "$FD_H" --summary-tol "$TOL" \
  --refine --refine-iters "$REFINE_ITERS"

run_scan \
  lagged a "$OUT_DIR/lagged_scan_a.txt" \
  --mode lagged --scan a \
  --a "$A0" --b "$B0" \
  --a-min "$A_MIN" --a-max "$A_MAX" --n "$N" \
  --selection-strength "$K" --fd-h "$FD_H" --summary-tol "$TOL" \
  --refine --refine-iters "$REFINE_ITERS"

run_scan \
  ode b "$OUT_DIR/ode_scan_b.txt" \
  --mode ode --scan b \
  --a "$A0" --b "$B0" \
  --b-min "$B_MIN" --b-max "$B_MAX" --n "$N" \
  --fd-h "$FD_H" --summary-tol "$TOL" \
  --refine --refine-iters "$REFINE_ITERS"

run_scan \
  ode a "$OUT_DIR/ode_scan_a.txt" \
  --mode ode --scan a \
  --a "$A0" --b "$B0" \
  --a-min "$A_MIN" --a-max "$A_MAX" --n "$N" \
  --fd-h "$FD_H" --summary-tol "$TOL" \
  --refine --refine-iters "$REFINE_ITERS"

echo "DONE"
echo "Results: $OUT_DIR"
echo "Review closest boundary candidates in:"
echo "  $OUT_DIR/lagged_scan_b.txt"
echo "  $OUT_DIR/lagged_scan_a.txt"
echo "  $OUT_DIR/ode_scan_b.txt"
echo "  $OUT_DIR/ode_scan_a.txt"
