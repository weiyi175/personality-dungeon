# B4 Short Scout Decision

## Conditions

- beta0p00_k0p08: beta_state_k=0.000000 k_base=0.080000 level3_seed_count=0 stage3_uplift=0.000000 gamma_ratio=1.000000 k_clamped_ratio=0.000000 short_scout_pass=no hard_stop_fail=no verdict=control simplex_png=reports/experiments/l3_bottleneck/phase2/b4_scout/beta0p00_k0p08/seed51_simplex.png
- beta0p60_k0p08: beta_state_k=0.600000 k_base=0.080000 level3_seed_count=0 stage3_uplift=-0.004476 gamma_ratio=2.457627 k_clamped_ratio=0.000000 short_scout_pass=no hard_stop_fail=yes verdict=fail simplex_png=reports/experiments/l3_bottleneck/phase2/b4_scout/beta0p60_k0p08/seed52_simplex.png

## Recommendation

- longer_confirm_candidate: none

## Stop Rule

- 若 0/3 seeds 達 Level 3 且 mean_stage3_score uplift < 0.02，該 cell 直接記為 fail，不做 Longer Confirm。
- 若 k_clamped_ratio > 0.3，視為高 clamp 飽和；若 > 0.5，視為嚴重飽和，可在同一 B4 family 內升級到更寬 clamp 或 exponential。
