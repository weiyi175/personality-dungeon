#!/bin/bash
# Phase 14: A組 seed=45+sampled+apv2加強 & B組 seed 46-50 sampled
set -e
cd /home/user/personality-dungeon
EVENTS_JSON=docs/personality_dungeon_v1/02_event_templates_v1.json

echo "=== Phase 14 A組: seed=45 sampled adaptive_payoff_v2 加強版 ==="
for s in 1.5 2.0 2.5; do
  for i in 300 400; do
    nohup ./venv/bin/python -m simulation.run_simulation \
      --enable-events --events-json "$EVENTS_JSON" \
      --popularity-mode sampled \
      --seed 45 --rounds 8000 --players 300 \
      --payoff-mode matrix_ab --a 0.8 --b 0.9 \
      --selection-strength 0.06 --init-bias 0.12 \
      --event-failure-threshold 0.72 --event-health-penalty 0.10 \
      --adaptive-payoff-strength "$s" \
      --payoff-update-interval "$i" \
      --adaptive-payoff-target 0.27 \
      --out "outputs/sampled_s45_apv2_s${s}_i${i}.csv" \
      > "outputs/_log_s45_apv2_s${s}_i${i}.txt" 2>&1 &
    echo "Launched A: strength=$s interval=$i pid=$!"
  done
done

echo "=== Phase 14 B組: seed 46-50 純 sampled ==="
for seed in 46 47 48 49 50; do
  nohup ./venv/bin/python -m simulation.run_simulation \
    --enable-events --events-json "$EVENTS_JSON" \
    --popularity-mode sampled \
    --seed "$seed" --rounds 8000 --players 300 \
    --payoff-mode matrix_ab --a 0.8 --b 0.9 \
    --selection-strength 0.06 --init-bias 0.12 \
    --event-failure-threshold 0.72 --event-health-penalty 0.10 \
    --out "outputs/sampled_seed${seed}.csv" \
    > "outputs/_log_sampled_seed${seed}.txt" 2>&1 &
  echo "Launched B: seed=$seed pid=$!"
done

echo "=== 全部 11 個模擬已啟動，等待完成 ==="
wait
echo "=== 全部完成 ==="
