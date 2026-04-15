#!/usr/bin/env bash
set -euo pipefail

# A-line refine experiment around the best structural-search candidate,
# with a different noise structure: popularity-mode=expected.
#
# Default grid:
#   b = 0.9
#   a in {1.04, 1.052, 1.064, 1.076, 1.088, 1.10}
#   popularity-mode = expected
#
# Default validation scale:
#   seeds = {45, 47, 49, 51}
#   rounds = 8000
#
# Usage:
#   ./scripts/run_hopf_aline_refine_expected.sh
#   SEEDS="45 47 49 51 53 55" J_MAX=12 ./scripts/run_hopf_aline_refine_expected.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not found: $PYTHON_BIN" >&2
  exit 1
fi

export POPULARITY_MODE="${POPULARITY_MODE:-expected}"
export A_GRID="${A_GRID:-1.04 1.052 1.064 1.076 1.088 1.10}"
export B_FIXED="${B_FIXED:-0.9}"
export SEEDS="${SEEDS:-45 47 49 51}"
export ROUNDS="${ROUNDS:-8000}"
export J_MAX="${J_MAX:-8}"
export RUN_TAG="${RUN_TAG:-hopf_aline_refine_expected_$(date +%Y%m%d_%H%M%S)}"

exec "$ROOT_DIR/scripts/run_hopf_candidates_a_line.sh"