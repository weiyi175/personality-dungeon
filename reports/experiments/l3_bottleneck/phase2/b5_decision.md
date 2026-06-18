# B5 Tangential Drift Decision

## G1 Deterministic Gate

- g1_mean_field_delta0p000: delta=0.000000 turn_ratio_vs_control=1.000000 drift_contribution_ratio=0.000000 g1_level_gate_pass=yes g1_turn_gate_pass=yes g1_drift_gate_pass=yes g1_gate_pass=yes verdict=control
- g1_mean_field_delta0p003: delta=0.003000 turn_ratio_vs_control=1.000000 drift_contribution_ratio=0.003572 g1_level_gate_pass=yes g1_turn_gate_pass=yes g1_drift_gate_pass=yes g1_gate_pass=yes verdict=pass
- g1_mean_field_delta0p006: delta=0.006000 turn_ratio_vs_control=1.000000 drift_contribution_ratio=0.007110 g1_level_gate_pass=yes g1_turn_gate_pass=yes g1_drift_gate_pass=yes g1_gate_pass=yes verdict=pass
- g1_mean_field_delta0p010: delta=0.010000 turn_ratio_vs_control=inf drift_contribution_ratio=0.011820 g1_level_gate_pass=yes g1_turn_gate_pass=yes g1_drift_gate_pass=yes g1_gate_pass=yes verdict=pass
- g1_mean_field_delta0p015: delta=0.015000 turn_ratio_vs_control=inf drift_contribution_ratio=0.017800 g1_level_gate_pass=yes g1_turn_gate_pass=yes g1_drift_gate_pass=yes g1_gate_pass=yes verdict=pass

## G2 Short Scout

- g2_sampled_delta0p000: delta=0.000000 g1_gate_pass=yes level3_seed_count=0 stage3_uplift=0.000000 mean_drift_norm=0.000000 mean_effective_delta_growth_ratio=0.000000 mean_tangential_alignment=0.000000 phase_amplitude_stability=0.975688 drift_rose_png=reports/experiments/l3_bottleneck/phase2/b5_scout/g2_sampled_delta0p000/seed45_drift_vector_rose.png short_scout_pass=no hard_stop_fail=no verdict=control
- g2_sampled_delta0p003: delta=0.003000 g1_gate_pass=yes level3_seed_count=0 stage3_uplift=0.000265 mean_drift_norm=0.003000 mean_effective_delta_growth_ratio=0.004677 mean_tangential_alignment=0.023852 phase_amplitude_stability=0.975674 drift_rose_png=reports/experiments/l3_bottleneck/phase2/b5_scout/g2_sampled_delta0p003/seed45_drift_vector_rose.png short_scout_pass=no hard_stop_fail=yes verdict=fail
- g2_sampled_delta0p006: delta=0.006000 g1_gate_pass=yes level3_seed_count=0 stage3_uplift=0.000215 mean_drift_norm=0.006000 mean_effective_delta_growth_ratio=0.009353 mean_tangential_alignment=0.024418 phase_amplitude_stability=0.975665 drift_rose_png=reports/experiments/l3_bottleneck/phase2/b5_scout/g2_sampled_delta0p006/seed45_drift_vector_rose.png short_scout_pass=no hard_stop_fail=yes verdict=fail
- g2_sampled_delta0p010: delta=0.010000 g1_gate_pass=yes level3_seed_count=0 stage3_uplift=0.000624 mean_drift_norm=0.010000 mean_effective_delta_growth_ratio=0.015587 mean_tangential_alignment=0.024497 phase_amplitude_stability=0.975639 drift_rose_png=reports/experiments/l3_bottleneck/phase2/b5_scout/g2_sampled_delta0p010/seed45_drift_vector_rose.png short_scout_pass=no hard_stop_fail=yes verdict=fail
- g2_sampled_delta0p015: delta=0.015000 g1_gate_pass=yes level3_seed_count=0 stage3_uplift=0.000459 mean_drift_norm=0.015000 mean_effective_delta_growth_ratio=0.023378 mean_tangential_alignment=0.024153 phase_amplitude_stability=0.975617 drift_rose_png=reports/experiments/l3_bottleneck/phase2/b5_scout/g2_sampled_delta0p015/seed45_drift_vector_rose.png short_scout_pass=no hard_stop_fail=no verdict=weak_positive

## Recommendation

- longer_confirm_candidate: none

## Stop Rule

- G1: 若 turn_strength < 0.92 × baseline、cycle level 掉出 3、或 drift_contribution_ratio > 0.05，該 delta 不得進 G2。
- G2: 若 delta <= 0.010 的 active cells 全部 0/3 Level 3 且 mean_stage3_score uplift < 0.02，B5 直接 closure。
- overall_verdict: close_b5
