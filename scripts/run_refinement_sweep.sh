#!/usr/bin/env bash
set -euo pipefail

# Refinement sweep runner (crossing band densification)
# - Reads k50 centers from a summary CSV
# - Runs one sweep CSV per N (so each N can have its own band)
# - Writes to a NEW output filename tag to avoid mixing CSV schemas/headers

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
SUMMARY_CSV="${SUMMARY_CSV:-outputs/analysis/rho_curve/rho_curve_merged_with_lowband_clean_plus_refine_summary.csv}"

# Which Ns to run
TARGETS="${TARGETS:-100 200 300 500 1000}"

# Crossing band config
STEP="${STEP:-0.005}"
BAND="${BAND:-0.05}"

# Protocol lock (recommended by SDD 4.6.9)
ROUNDS="${ROUNDS:-4000}"
BURN_FRAC="${BURN_FRAC:-0.30}"
TAIL="${TAIL:-1000}"
SEEDS_SPEC="${SEEDS_SPEC:-0:29}"
JOBS="${JOBS:-0}"
SERIES="${SERIES:-p}"

# Model / pipeline knobs
PAYOFF_MODE="${PAYOFF_MODE:-matrix_ab}"
A_VAL="${A_VAL:-0.4}"
B_VAL="${B_VAL:-0.2425406997}"
GAMMA_VAL="${GAMMA_VAL:-0.1}"
EPSILON_VAL="${EPSILON_VAL:-0.0}"
POPULARITY_MODE="${POPULARITY_MODE:-sampled}"
EVOLUTION_MODE="${EVOLUTION_MODE:-sampled}"
PAYOFF_LAG="${PAYOFF_LAG:-1}"
INIT_BIAS="${INIT_BIAS:-0.0}"

# Level thresholds
AMP_THR="${AMP_THR:-0.02}"
MIN_LAG="${MIN_LAG:-2}"
MAX_LAG="${MAX_LAG:-500}"
CORR_THR="${CORR_THR:-0.09}"
ETA="${ETA:-0.55}"
STAGE3_METHOD="${STAGE3_METHOD:-turning}"
PHASE_SMOOTHING="${PHASE_SMOOTHING:-1}"

# Output naming tag: avoid mixing old/new CSV schemas.
TAG="${TAG:-prov2_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/sweeps/rho_curve}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "ERROR: SUMMARY_CSV not found: $SUMMARY_CSV" >&2
  exit 2
fi

# Generate (N, kmin, kmax) triples from summary.csv
mapfile -t TRIPLES < <(
  SUMMARY_CSV="$SUMMARY_CSV" \
  TARGETS="$TARGETS" \
  STEP="$STEP" \
  BAND="$BAND" \
  "$PYTHON_BIN" - <<'PY'
import csv
import os
import sys

summary_path = os.environ["SUMMARY_CSV"]
targets = [int(x) for x in os.environ.get("TARGETS", "").split() if x.strip()]
step = float(os.environ.get("STEP", "0.005"))
band = float(os.environ.get("BAND", "0.05"))

if not targets:
  raise SystemExit("TARGETS is empty")

k50_by_n = {}
with open(summary_path, newline="") as f:
  for row in csv.DictReader(f):
    try:
      n = int(float(row.get("players") or ""))
    except Exception:
      continue
    k50 = row.get("k50")
    if not k50:
      continue
    try:
      k50_by_n[n] = float(k50)
    except Exception:
      pass

missing = [n for n in targets if n not in k50_by_n]
if missing:
  raise SystemExit(f"Missing k50 for N={missing} in {summary_path}")

for n in targets:
  c = k50_by_n[n]
  # round center to the grid
  c = round(c / step) * step
  kmin = c - band
  kmax = c + band
  sys.stdout.write(f"{n} {kmin:.3f} {kmax:.3f}\n")
PY
)

echo "=== refinement sweep (crossing densification) ==="
echo "summary=$SUMMARY_CSV"
echo "targets=$TARGETS"
echo "step=$STEP band=$BAND rounds=$ROUNDS burn_frac=$BURN_FRAC tail=$TAIL seeds=$SEEDS_SPEC"
echo "tag=$TAG out_dir=$OUT_DIR"
echo

# Derive a human-readable seed count for filenames/logging.
SEED_COUNT="$(
  SEEDS_SPEC="$SEEDS_SPEC" \
  "$PYTHON_BIN" - <<'PY'
import os

s = str(os.environ.get('SEEDS_SPEC', '')).strip()
if not s:
  raise SystemExit(2)

def parse_seeds(spec: str) -> list[int]:
  spec = spec.strip()
  if ',' in spec:
    out = []
    for part in spec.split(','):
      part = part.strip()
      if part:
        out.append(int(part))
    return out
  if ':' in spec:
    parts = [p.strip() for p in spec.split(':')]
    if len(parts) not in (2, 3):
      raise ValueError('bad range')
    start = int(parts[0]); end = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else 1
    if step == 0:
      raise ValueError('step=0')
    if step > 0:
      return list(range(start, end + 1, step))
    return list(range(start, end - 1, step))
  return [int(spec)]

print(len(parse_seeds(s)))
PY
)"

for triple in "${TRIPLES[@]}"; do
  read -r N KMIN KMAX <<<"$triple"
  OUT="$OUT_DIR/rho_curve_refine_crossing_${TAG}_a0p4_b0p2425407_lag1_eta055_N${N}_k${KMIN}_${KMAX}_s${STEP}_R${ROUNDS}_S${SEED_COUNT}_tail${TAIL}.csv"

  echo "--- N=$N k=${KMIN}:${KMAX}:${STEP} ---"
  echo "out=$OUT"

  "$PYTHON_BIN" -m simulation.rho_curve \
    --payoff-mode "$PAYOFF_MODE" --a "$A_VAL" --b "$B_VAL" \
    --gamma "$GAMMA_VAL" --epsilon "$EPSILON_VAL" \
    --popularity-mode "$POPULARITY_MODE" --evolution-mode "$EVOLUTION_MODE" --payoff-lag "$PAYOFF_LAG" \
    --init-bias "$INIT_BIAS" \
    --players-grid "$N" \
    --rounds "$ROUNDS" --burn-in-frac "$BURN_FRAC" --tail "$TAIL" \
    --seeds "$SEEDS_SPEC" --jobs "$JOBS" \
    --series "$SERIES" \
    --k-grid "${KMIN}:${KMAX}:${STEP}" \
    --amplitude-threshold "$AMP_THR" --min-lag "$MIN_LAG" --max-lag "$MAX_LAG" --corr-threshold "$CORR_THR" \
    --eta "$ETA" --stage3-method "$STAGE3_METHOD" --phase-smoothing "$PHASE_SMOOTHING" \
    --resume \
    --out "$OUT"
done

echo "DONE: refinement sweeps wrote into $OUT_DIR"
