#!/usr/bin/env bash
set -euo pipefail

# N-expansion + refinement-first runner
#
# Goal:
# - Add MORE system sizes N (players) beyond the current set.
# - Increase statistical power (more seeds) and k-resolution near crossing.
#
# Pipeline:
# 1) Coarse sweep over new N set (one CSV, broad k-grid).
# 2) Scaling summary (bootstrap + Bayes) to estimate k50 for those N.
# 3) Refinement sweep per N (crossing band densification) using scripts/run_refinement_sweep.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
OUT_DIR="${OUT_DIR:-outputs/sweeps/rho_curve}"
ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/analysis/rho_curve}"
mkdir -p "$OUT_DIR" "$ANALYSIS_DIR"

# Tag to avoid mixing outputs with previous sweeps.
TAG="${TAG:-prov3_nexp_$(date +%Y%m%d_%H%M%S)}"

# -------------
# 1) Coarse sweep (new Ns)
# -------------
# Example targets (override):
#   COARSE_TARGETS="75,125,150,250,400,750,1500"
COARSE_TARGETS="${COARSE_TARGETS:-75,125,150,250,400,750,1500}"
COARSE_K_GRID="${COARSE_K_GRID:-0.2:1.2:0.02}"
COARSE_ROUNDS="${COARSE_ROUNDS:-4000}"
COARSE_BURN_FRAC="${COARSE_BURN_FRAC:-0.30}"
COARSE_TAIL="${COARSE_TAIL:-1000}"
COARSE_SEEDS_SPEC="${COARSE_SEEDS_SPEC:-0:29}"
COARSE_JOBS="${COARSE_JOBS:-0}"
COARSE_SERIES="${COARSE_SERIES:-p}"

# Model / pipeline knobs (keep aligned with existing rho_curve sweeps unless explicitly changed)
PAYOFF_MODE="${PAYOFF_MODE:-matrix_ab}"
A_VAL="${A_VAL:-0.4}"
B_VAL="${B_VAL:-0.2425406997}"
GAMMA_VAL="${GAMMA_VAL:-0.1}"
EPSILON_VAL="${EPSILON_VAL:-0.0}"
POPULARITY_MODE="${POPULARITY_MODE:-sampled}"
EVOLUTION_MODE="${EVOLUTION_MODE:-sampled}"
PAYOFF_LAG="${PAYOFF_LAG:-1}"
INIT_BIAS="${INIT_BIAS:-0.0}"

AMP_THR="${AMP_THR:-0.02}"
MIN_LAG="${MIN_LAG:-2}"
MAX_LAG="${MAX_LAG:-500}"
CORR_THR="${CORR_THR:-0.09}"
ETA="${ETA:-0.55}"
STAGE3_METHOD="${STAGE3_METHOD:-turning}"
PHASE_SMOOTHING="${PHASE_SMOOTHING:-1}"

COARSE_OUT="$OUT_DIR/rho_curve_coarse_${TAG}_a0p4_b0p2425407_lag1_eta055_N${COARSE_TARGETS//,/}_k${COARSE_K_GRID//:/_}_R${COARSE_ROUNDS}_S${COARSE_SEEDS_SPEC//:/_}.csv"

echo "=== [1/3] coarse sweep (new N) ==="
echo "tag=$TAG"
echo "players_grid=$COARSE_TARGETS"
echo "k_grid=$COARSE_K_GRID"
echo "rounds=$COARSE_ROUNDS burn_frac=$COARSE_BURN_FRAC tail=$COARSE_TAIL seeds=$COARSE_SEEDS_SPEC jobs=$COARSE_JOBS"
echo "out=$COARSE_OUT"
echo

"$PYTHON_BIN" -m simulation.rho_curve \
  --payoff-mode "$PAYOFF_MODE" --a "$A_VAL" --b "$B_VAL" \
  --gamma "$GAMMA_VAL" --epsilon "$EPSILON_VAL" \
  --popularity-mode "$POPULARITY_MODE" --evolution-mode "$EVOLUTION_MODE" --payoff-lag "$PAYOFF_LAG" \
  --init-bias "$INIT_BIAS" \
  --players-grid "$COARSE_TARGETS" \
  --rounds "$COARSE_ROUNDS" --burn-in-frac "$COARSE_BURN_FRAC" --tail "$COARSE_TAIL" \
  --seeds "$COARSE_SEEDS_SPEC" --jobs "$COARSE_JOBS" \
  --series "$COARSE_SERIES" \
  --k-grid "$COARSE_K_GRID" \
  --amplitude-threshold "$AMP_THR" --min-lag "$MIN_LAG" --max-lag "$MAX_LAG" --corr-threshold "$CORR_THR" \
  --eta "$ETA" --stage3-method "$STAGE3_METHOD" --phase-smoothing "$PHASE_SMOOTHING" \
  --resume \
  --out "$COARSE_OUT"

# -------------
# 2) Scaling summary for k50 centers
# -------------
SCALING_PREFIX="${SCALING_PREFIX:-rho_curve_${TAG}_bootbayes}"
BOOTSTRAP_RESAMPLES="${BOOTSTRAP_RESAMPLES:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-123}"
BAYES_DRAWS="${BAYES_DRAWS:-2000}"
BAYES_SEED="${BAYES_SEED:-123}"
BAYES_PRIOR_SIGMA="${BAYES_PRIOR_SIGMA:-10}"
BAYES_ASSUME_INCREASING="${BAYES_ASSUME_INCREASING:-1}"

# Default scaling inputs: reuse the current merged set + the new coarse sweep.
# You can override SCALING_INPUTS as a space-separated list of files.
SCALING_INPUTS_DEFAULT=(
  "outputs/sweeps/rho_curve/rho_curve_a0p4_b0p2425407_lag1_sampled_eta055_R4000_S30_N50_200_1000_k0p1_1p0_s0p02.csv"
  "outputs/sweeps/rho_curve/rho_curve_a0p4_b0p2425407_lag1_sampled_eta055_R4000_S30_N200_k1p0_1p4_s0p02.csv"
  "outputs/sweeps/rho_curve/rho_curve_critical_band_a0p4_b0p2425407_lag1_eta055_R2500_S10_N100_300_500_2000_k0p7_1p2_s0p02.csv"
  "outputs/sweeps/rho_curve/rho_curve_lowband_corr0p09_clean_N300_500.csv"
  "$COARSE_OUT"
)

if [[ -n "${SCALING_INPUTS:-}" ]]; then
  # shellcheck disable=SC2206
  SCALING_INPUTS_ARR=( $SCALING_INPUTS )
else
  SCALING_INPUTS_ARR=("${SCALING_INPUTS_DEFAULT[@]}")
fi

SCALING_ARGS=()
for p in "${SCALING_INPUTS_ARR[@]}"; do
  if [[ -f "$p" ]]; then
    SCALING_ARGS+=(--in "$p")
  else
    echo "WARN: scaling input missing, skip: $p" >&2
  fi
done

echo
echo "=== [2/3] scaling (bootstrap + bayes) ==="
echo "prefix=$SCALING_PREFIX outdir=$ANALYSIS_DIR"

"$PYTHON_BIN" -m analysis.rho_curve_scaling \
  "${SCALING_ARGS[@]}" \
  --outdir "$ANALYSIS_DIR" \
  --prefix "$SCALING_PREFIX" \
  --bootstrap-resamples "$BOOTSTRAP_RESAMPLES" \
  --bootstrap-seed "$BOOTSTRAP_SEED" \
  --bayes-method laplace \
  --bayes-draws "$BAYES_DRAWS" \
  --bayes-seed "$BAYES_SEED" \
  --bayes-prior-sigma "$BAYES_PRIOR_SIGMA" \
  --bayes-assume-increasing "$BAYES_ASSUME_INCREASING"

SUMMARY_CSV="$ANALYSIS_DIR/${SCALING_PREFIX}_summary.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "ERROR: expected scaling summary not found: $SUMMARY_CSV" >&2
  exit 2
fi

# -------------
# 3) Refinement sweep per N
# -------------
# By default, refine exactly the new N set.
REFINE_TARGETS="${REFINE_TARGETS:-${COARSE_TARGETS//,/ }}"
REFINE_STEP="${REFINE_STEP:-0.0025}"
REFINE_BAND="${REFINE_BAND:-0.05}"
REFINE_SEEDS_SPEC="${REFINE_SEEDS_SPEC:-0:59}"

echo
echo "=== [3/3] refinement sweep (per N) ==="
echo "summary=$SUMMARY_CSV"
echo "targets=$REFINE_TARGETS step=$REFINE_STEP band=$REFINE_BAND seeds=$REFINE_SEEDS_SPEC"
echo

SUMMARY_CSV="$SUMMARY_CSV" \
TARGETS="$REFINE_TARGETS" \
STEP="$REFINE_STEP" \
BAND="$REFINE_BAND" \
ROUNDS="$COARSE_ROUNDS" \
BURN_FRAC="$COARSE_BURN_FRAC" \
TAIL="$COARSE_TAIL" \
SEEDS_SPEC="$REFINE_SEEDS_SPEC" \
JOBS="$COARSE_JOBS" \
SERIES="$COARSE_SERIES" \
PAYOFF_MODE="$PAYOFF_MODE" A_VAL="$A_VAL" B_VAL="$B_VAL" GAMMA_VAL="$GAMMA_VAL" EPSILON_VAL="$EPSILON_VAL" \
POPULARITY_MODE="$POPULARITY_MODE" EVOLUTION_MODE="$EVOLUTION_MODE" PAYOFF_LAG="$PAYOFF_LAG" INIT_BIAS="$INIT_BIAS" \
AMP_THR="$AMP_THR" MIN_LAG="$MIN_LAG" MAX_LAG="$MAX_LAG" CORR_THR="$CORR_THR" ETA="$ETA" STAGE3_METHOD="$STAGE3_METHOD" PHASE_SMOOTHING="$PHASE_SMOOTHING" \
TAG="$TAG" OUT_DIR="$OUT_DIR" PYTHON_BIN="$PYTHON_BIN" \
./scripts/run_refinement_sweep.sh

echo "DONE: N-expansion + refinement. tag=$TAG"