# P7-H Power Analysis under Realistic Human Noise

**Date:** 2026-06-06   **Script:** `scripts/experiments/power_p7h_human_noise.py`
**Primary DV:** `max_proximity`   **Planned N:** 106 / group (212 total)

## Question

The simulated apparatus produces a huge objective effect (max_proximity d≈3.2
feedback-ON, capped regime). Real humans are noisier and won't all engage with
the manipulation. **Is 106/group safe, and what actually drives the required N?**

## Model

Ground truth = the noiseless simulated per-group `max_proximity` distributions
(feedback ON, capped 4-event regime; base exp=0.958±0.034, ctrl=0.525±0.189,
raw d=3.19, from 400 sessions). Two independent human degradations:

- **engagement attenuation `a`** = P(an experiment player responds to the
  manipulation); non-responders' DV is drawn from the control distribution.
- **individual noise `σ`** = additive Gaussian on max_proximity, clipped to [0,1].

Monte-Carlo (1500 reps): power at N=106 and the minimum N for 80% power.

## H1 results — power @ N=106 / min N for 80%

| attenuation | σ=0.00 | σ=0.15 | σ=0.30 |
|-------------|--------|--------|--------|
| a=1.00 (all respond) | 100% / N≥10 | 100% / N≥10 | 100% / N≥10 |
| a=0.60 | 100% / N≥15 | 100% / N≥15 | 100% / N≥30 |
| a=0.40 | 100% / N≥25 | 100% / N≥40 | 94% / N≥80 |
| a=0.25 (1-in-4 respond) | 97% / N≥64 | 86% / N≥106 | 62% / N≥212 |

**H1 is robust.** 106/group keeps ≥94% power unless engagement is very poor
(≤25% of experiment players respond) *and* noise is high — a pessimistic corner.
Objective H1 is not the sample-size constraint.

## H2 results — analytic power by assumed UX effect size

| true d (subjective) | power @ 106 | N/group for 80% |
|---------------------|-------------|-----------------|
| 0.2 (small) | 31% | 392 |
| 0.3 | 59% | 174 |
| 0.5 (medium) | 95% | 63 |
| 0.8 (large) | 100% | 25 |

**H2 is the binding constraint.** If the true subjective UX effect is small
(d ≤ 0.3), 106/group is underpowered (≤59%). 106/group is only comfortable if the
UX effect is medium or larger (d ≥ 0.5).

## Recommendation

1. **Keep N = 106/group as planned for H1** — the objective effect is over-
   powered there even under heavy human attenuation/noise.
2. **Treat H2 as the sample-size driver and decide its target effect size.**
   - If a *medium* UX effect (d≥0.5) is the smallest worth detecting: 106/group
     gives 95% power — fine.
   - If a *small* UX effect (d≈0.3) must be detectable: plan ~**174/group**
     (≈350 total), or pre-register H2 as exploratory and report the achieved
     power/CI rather than a go/no-go significance test.
3. **Track engagement during collection.** If observed experiment-arm
   `max_proximity` is far below the simulated ~0.95 (i.e. real attenuation worse
   than a≈0.4), revisit H1 power mid-stream (without peeking at the effect — use
   the proximity *distribution*, not the group contrast).

## Reproduce

```bash
./venv/bin/python scripts/experiments/power_p7h_human_noise.py \
    --planned-n 106 --reps 1500 --base-sessions 400 \
    --out reports/experiments/p7h_power
```

> Caveat: the H1 base distribution comes from the agent-population simulator, not
> humans; it is the best available pre-collection estimate. The attenuation/noise
> grid brackets plausible human degradation but the true values are unknown until
> data arrives — hence recommendation #3.
