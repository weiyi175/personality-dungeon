# C1 Local Pairwise Imitation Decision

## G2 Short Scout

- g2_c1_control: topo=none β=5.000000 μ=0.500000 level3=0 mean_entropy=1.093775 spatial_clust=0.000000 weight_std=0.000000 pass= verdict=control
- g2_c1_small_world_b10p0_m0p5: topo=small_world β=10.000000 μ=0.500000 level3=0 mean_entropy=1.093778 spatial_clust=1.000000 weight_std=0.000000 pass=no verdict=fail

## Recommendation

- longer_confirm_candidate: none

## Pass Gate (4-way AND)

1. level3_seed_count ≥ 2
2. mean_env_gamma ≥ 0
3. mean_player_weight_entropy ≥ 1.18
4. spatial_strategy_clustering > 0.25

## Stop Rule

- If ALL active conditions hard-stop fail, C1 closes as negative result.
- Interpretation: per-player pairwise Fermi imitation on structured graph is insufficient to break entropy lock.
- overall_verdict: close_c1
