#!/usr/bin/env bash
set -euo pipefail

# Diagnose P0 outputs with two tails (1000 and 3000), and apply pass rule.
#
# Pass rule (strict):
# - tail=3000: level >= 3 AND score > 0.55
#
# Usage:
#   ./scripts/diag_p0_stabilization.sh
#   ./scripts/diag_p0_stabilization.sh outputs/p0_stabilization/<RUN_TAG>
#   ./scripts/diag_p0_stabilization.sh outputs/p0_stabilization/<RUN_TAG> > /tmp/p0_diag.txt

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
DIAG_SCRIPT="${DIAG_SCRIPT:-$ROOT_DIR/scripts/diagnose_cycle.py}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$DIAG_SCRIPT" ]]; then
  echo "ERROR: diagnose script not found: $DIAG_SCRIPT" >&2
  exit 1
fi

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [p0_output_dir]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  RUN_DIR="$1"
else
  P0_ROOT="$ROOT_DIR/outputs/p0_stabilization"
  if [[ ! -d "$P0_ROOT" ]]; then
    echo "ERROR: P0 output root not found: $P0_ROOT" >&2
    echo "Run P0 first: ./scripts/run_p0_stabilization.sh" >&2
    echo "Or specify a run directory explicitly: $0 [p0_output_dir]" >&2
    exit 1
  fi

  LATEST_RUN="$(find "$P0_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [[ -z "$LATEST_RUN" ]]; then
    echo "ERROR: no P0 run directory found under $P0_ROOT" >&2
    echo "Run P0 first: ./scripts/run_p0_stabilization.sh" >&2
    echo "Or specify a run directory explicitly: $0 [p0_output_dir]" >&2
    exit 1
  fi
  RUN_DIR="$LATEST_RUN"
fi

MANIFEST="$RUN_DIR/manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest not found: $MANIFEST" >&2
  exit 1
fi

echo "RUN_DIR = $RUN_DIR"

TMP1="$(mktemp)"
TMP3="$(mktemp)"
trap 'rm -f "$TMP1" "$TMP3"' EXIT

mapfile -t CSVS < <(awk -F '\t' 'NR>1 {print $5}' "$MANIFEST")
if [[ ${#CSVS[@]} -eq 0 ]]; then
  echo "ERROR: no csv paths in manifest: $MANIFEST" >&2
  exit 1
fi

echo "=== tail=1000 ==="
"$PYTHON_BIN" "$DIAG_SCRIPT" --tail 1000 "${CSVS[@]}" | tee "$TMP1"

echo
echo "=== tail=3000 ==="
"$PYTHON_BIN" "$DIAG_SCRIPT" --tail 3000 "${CSVS[@]}" | tee "$TMP3"

echo
echo "=== PASS CHECK (tail=3000: level>=3 and score>0.55) ==="
awk -F '\t' '
{
  csv=$1; level=-1; score=-1; error="";
  for (i=2; i<=NF; i++) {
    if ($i ~ /^level=/) { split($i,a,"="); level=a[2]+0 }
    if ($i ~ /^score=/) { split($i,a,"="); score=a[2]+0 }
    if ($i ~ /^ERROR=/) { error=$i }
  }
  if (error != "") {
    printf "%s\t%s\tpass=NO\n", csv, error
    total += 1
    next
  }
  pass=(level>=3 && score>0.55)
  if (csv != "") {
    printf "%s\tlevel=%d\tscore=%.4f\tpass=%s\n", csv, level, score, (pass?"YES":"NO")
    total += 1
    if (pass) ok += 1
  }
}
END {
  printf "summary\tpass=%d/%d\n", ok, total
}
' "$TMP3"
