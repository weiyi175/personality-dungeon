#!/usr/bin/env bash
set -euo pipefail

# Runner for B1 smoke: two dispatch modes x six seeds
# Adjust CLI flags below if the module's CLI differs.

PY=./venv/bin/python
SEEDS="42,44,45,67,73,90"
MODES=("async_round_robin" "async_poisson")
WORLD_FEEDBACK="adaptive_world"
EVENTS_JSON="docs/personality_dungeon_v1/02_event_templates_smoke_v1.json"
DISPATCH_RATE="0.08"
OUT_DIR="outputs/b1_smoke"

mkdir -p "$OUT_DIR"

for mode in "${MODES[@]}"; do
  mode_dir="$OUT_DIR/${mode}"
  mkdir -p "$mode_dir"

  echo "[B1 SMOKE] mode=$mode seeds=$SEEDS (personality_rl smoke-only)"
  mode_out_base="$mode_dir/personality_rl"
  mkdir -p "$mode_out_base"

  IFS=',' read -ra seed_list <<< "$SEEDS"
  for s in "${seed_list[@]}"; do
    echo "  -> running seed=$s"
    "$PY" -m simulation.personality_rl_runtime \
      --seeds "$s" \
      --event-dispatch-mode "$mode" \
      --world-feedback-mode "$WORLD_FEEDBACK" \
      --events-json "$EVENTS_JSON" \
      --event-dispatch-target-rate "$DISPATCH_RATE" \
      --out-dir "$mode_out_base/seed_$s"
  done

done

echo "All runs completed. Outputs in $OUT_DIR" 
