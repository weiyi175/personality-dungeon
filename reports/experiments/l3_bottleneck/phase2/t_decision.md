# T-series: Topology × Update 2×2 — Short Scout Decision

## G2 Short Scout

condition: control_random_init  init=random_init  seeds=10  lv3=0  gamma=-0.000013  pa_stab=0.665529  iwd=0.374817  cosine=  pgd=  bps=  sac_d1=  sac_d2=  verdict=control
condition: control_uniform_init  init=uniform_init  seeds=10  lv3=0  gamma=0.000007  pa_stab=0.462013  iwd=0.000000  cosine=  pgd=  bps=  sac_d1=  sac_d2=  verdict=control
condition: t_lattice4_minibatch  graph=lattice4  update=minibatch  init=random_init  seeds=10  lv3=0  stage3_uplift=0.003129  gamma=-0.000009  pa_stab=0.708790  iwd=0.374817  cosine=0.085388  pgd=0.265495  bps=1.556766  sac_d1=-0.012358  sac_d2=0.001945  verdict=weak_positive

## Mechanism Signal

- t_lattice4_minibatch: iwd=0.3748  bps=1.5568 rad  sac_d1=-0.012358  sac_d2=0.001945  decay_ratio=-6.35

## Overall Verdict

- overall_verdict: weak_positive_t
- 建議 targeted follow-up（最多 9 runs）
