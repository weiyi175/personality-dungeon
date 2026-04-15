#!/usr/bin/env bash
set -euo pipefail

# Long-run confirmation for shortlisted LV3 candidates.
#
# Why this script exists:
# - lv3_verify_and_refine.sh is now the exploration script.
# - This script is the exploitation/confirmation script.
# - It fixes c=0.08 and evaluates a shortlist of (strength, interval)
#   combinations with more seeds and longer rounds, then ranks them by
#   average diagnosed level/score.
#
# Defaults are intentionally narrow around the current best-known basin.
# Override via env vars when needed.
#
# Examples:
#   ./scripts/run_lv3_candidate_confirm.sh
#   CANDIDATES="1.0:100 1.1:100 1.0:80" ROUNDS="10000 12000" ./scripts/run_lv3_candidate_confirm.sh

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

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

J_MAX="${J_MAX:-6}"
RUN_TAG="${RUN_TAG:-lv3_confirm_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/lv3_confirm/$RUN_TAG}"
LOG_DIR="${LOG_DIR:-logs/$RUN_TAG}"

FIXED_CROSS="${FIXED_CROSS:-0.08}"
FIXED_TARGET="${FIXED_TARGET:-0.27}"
SEEDS="${SEEDS:-45 47 49 51 53 55 57 59}"
ROUNDS="${ROUNDS:-10000 12000}"
CANDIDATES="${CANDIDATES:-1.0:100 1.1:100 1.0:80 1.2:100}"

EVENTS_JSON="docs/personality_dungeon_v1/02_event_templates_v1.json"
COMMON_FLAGS=(
  --enable-events --events-json "$EVENTS_JSON"
  --popularity-mode sampled
  --players 300
  --payoff-mode matrix_ab --a 0.8 --b 0.9
  --selection-strength 0.06 --init-bias 0.12
  --event-failure-threshold 0.72 --event-health-penalty 0.10
)

mkdir -p "$OUT_DIR" "$LOG_DIR"

ACTIVE_PIDS=()
ALL_OUTS=()
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
  local label="$1" out="$2"; shift 2
  wait_for_slot
  local logfile="$LOG_DIR/_$(basename "$out" .csv).log"
  nohup "$PYTHON_BIN" -m simulation.run_simulation \
    "${COMMON_FLAGS[@]}" "$@" --out "$out" \
    > "$logfile" 2>&1 &
  local pid=$!
  ACTIVE_PIDS+=("$pid")
  ALL_OUTS+=("$out")
  PID_LABEL[$pid]="$label"
  PID_LOG[$pid]="$logfile"
  printf "  [LAUNCH] %s\n" "$label"
}

echo "ROOT_DIR    = $ROOT_DIR"
echo "PYTHON_BIN  = $PYTHON_BIN"
echo "DIAG_SCRIPT = $DIAG_SCRIPT"
echo "J_MAX       = $J_MAX"
echo "OUT_DIR     = $OUT_DIR"
echo "LOG_DIR     = $LOG_DIR"
echo "FIXED_CROSS = $FIXED_CROSS"
echo "FIXED_TARGET= $FIXED_TARGET"
echo "SEEDS       = $SEEDS"
echo "ROUNDS      = $ROUNDS"
echo "CANDIDATES  = $CANDIDATES"
echo "--------------------------------------------------"

printf "out\tseed\trounds\tcross\tstrength\tinterval\ttarget\n" > "$OUT_DIR/manifest.tsv"

read -r -a SEED_ARR <<< "$SEEDS"
read -r -a ROUND_ARR <<< "$ROUNDS"
read -r -a CANDIDATE_ARR <<< "$CANDIDATES"

echo "=== Launch confirmation jobs ==="
for candidate in "${CANDIDATE_ARR[@]}"; do
  strength="${candidate%%:*}"
  interval="${candidate##*:}"
  strength_tag="$(printf "%s" "$strength" | sed 's/\.//g')"
  for rounds in "${ROUND_ARR[@]}"; do
    for seed in "${SEED_ARR[@]}"; do
      out="$OUT_DIR/confirm_seed${seed}_c0p08_s${strength_tag}_i${interval}_r${rounds}.csv"
      label="s=${strength} i=${interval} r=${rounds} seed=${seed}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$out" "$seed" "$rounds" "$FIXED_CROSS" "$strength" "$interval" "$FIXED_TARGET" >> "$OUT_DIR/manifest.tsv"
      launch_sim "$label" "$out" \
        --seed "$seed" \
        --rounds "$rounds" \
        --matrix-cross-coupling "$FIXED_CROSS" \
        --adaptive-payoff-strength "$strength" \
        --payoff-update-interval "$interval" \
        --adaptive-payoff-target "$FIXED_TARGET"
    done
  done
done

echo
echo "=== Waiting for simulations ==="
wait_for_all

if [[ $FAILED -gt 0 ]]; then
  echo "WARN: $FAILED simulation jobs failed; continuing with available CSVs" >&2
fi

echo
echo "=== Running diagnosis ==="
SUMMARY_TSV="$OUT_DIR/summary.tsv"
SUMMARY_SCRIPT="$OUT_DIR/_build_summary.py"

cat > "$SUMMARY_SCRIPT" <<'PY'
import csv
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST"])
summary_path = Path(os.environ["SUMMARY_TSV"])
python_bin = os.environ["PYTHON_BIN"]
diag_script = os.environ["DIAG_SCRIPT"]

diag_re = re.compile(r"level=(?P<level>\d+)\s+score=(?P<score>[0-9.]+)\s+turn=(?P<turn>[0-9.eE+-]+)")

records = []
with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        csv_path = Path(row["out"])
        if not csv_path.exists():
            continue
        proc = subprocess.run(
            [python_bin, diag_script, str(csv_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            continue
        match = diag_re.search(line[-1])
        if not match:
            continue
        records.append({
            **row,
            "level": int(match.group("level")),
            "score": float(match.group("score")),
            "turn": float(match.group("turn")),
        })

records.sort(
    key=lambda r: (
        float(r["strength"]),
        int(r["interval"]),
        int(r["rounds"]),
        int(r["seed"]),
    )
)

fieldnames = [
    "out", "seed", "rounds", "cross", "strength", "interval", "target",
    "level", "score", "turn",
]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(records)

print(f"Summary written to {summary_path}")
print()
print("=== Ranked configurations ===")
by_cfg = defaultdict(list)
for row in records:
    key = (row["strength"], row["interval"], row["rounds"])
    by_cfg[key].append(row)

ranked = []
for key, rows in by_cfg.items():
    avg_level = sum(r["level"] for r in rows) / len(rows)
    avg_score = sum(r["score"] for r in rows) / len(rows)
    min_level = min(r["level"] for r in rows)
    ranked.append((avg_level, avg_score, min_level, len(rows), key))

ranked.sort(reverse=True)
for avg_level, avg_score, min_level, n, key in ranked:
    print(
        f"s={key[0]} i={key[1]} r={key[2]} | "
        f"avg_level={avg_level:.3f} avg_score={avg_score:.4f} min_level={min_level} n={n}"
    )
PY

MANIFEST="$OUT_DIR/manifest.tsv" \
SUMMARY_TSV="$SUMMARY_TSV" \
PYTHON_BIN="$PYTHON_BIN" \
DIAG_SCRIPT="$DIAG_SCRIPT" \
"$PYTHON_BIN" "$SUMMARY_SCRIPT"

echo
echo "DONE"
echo "CSV     : $OUT_DIR"
echo "LOG     : $LOG_DIR"
echo "MANIFEST: $OUT_DIR/manifest.tsv"
echo "SUMMARY : $SUMMARY_TSV"
