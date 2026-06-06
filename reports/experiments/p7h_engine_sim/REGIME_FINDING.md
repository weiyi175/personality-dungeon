# P7-H Apparatus Validation — RL-Engine Space-A Event Path

**Date:** 2026-06-06
**Branch:** feature/phase4-todos
**Driver:** `scripts/experiments/run_p7h_engine_sim.py`
**Analysis:** `scripts/experiments/analyze_p7h_real_study.py` (pre-registered H1/H2/H3)

## What this validates

Unlike the standalone decay-model simulation (`run_p7h_player_test.py`), this
driver exercises the **real redesigned apparatus** the live Godot study will use:

- a 50-agent `RLSessionEngine` per session,
- events applied via `engine.apply_personality_event()` in **Space A**,
- the DV read from the engine's Space-A displacement
  (`snapshot.personality_displacement`),
- output written in the live-study JSON schema and fed to the pre-registered
  analysis.

This is the first end-to-end test of the Space A/B realignment
(`simulation/personality_space.py` + the engine event path, commit `8508266`)
that does **not** require Godot.

## Headline result (recommended operating regime)

Config: 60 sessions (30/30), 50 agents/session, **12 actions, event every 3
(→ 4 events)**, ΔP feedback off (isolates the event effect).

| group | n | mean displacement | mean max proximity | crossings |
|-------|---|-------------------|--------------------|-----------|
| control    | 30 | 0.07456 | 0.443 | 0.00 |
| experiment | 30 | 0.08874 | 0.943 | 1.00 |

Pre-registered H1 (Welch, one-sided, experiment > control):

- **t = 3.16, p = 1.37e-3 ✅ significant**
- **Cohen's d = 0.816, 95% CI [0.289, 1.343]**
- achieved power = 0.885; N/group for 80% power = 24

The redesigned apparatus produces the predicted direction with a large effect,
through the real engine path, and proximity no longer saturates instantly (the
original Space A/B coordinate-mismatch failure).

## Critical regime finding (affects pre-registration)

The effect sign depends on the event-sequence length, because event intensity is
**proximity-modulated** (`intensity_modulation`: closer to bifurcation → smaller
nudge). Once the experiment arm's proximity saturates (→1.0), its aligned events
decay to the min nudge and its net displacement plateaus, while the control arm
— which stays farther from critical — keeps receiving full-size nudges and
eventually overtakes it.

Sweep over `n_actions` (event_every=3, no feedback, N=60):

| n_actions | events | exp max prox | Cohen's d | verdict |
|-----------|--------|--------------|-----------|---------|
| 6  | 2 | 0.77 | +0.56 | clean exp>ctrl |
| 9  | 3 | 0.89 | +0.44 | clean |
| 12 | 4 | 0.94 | **+0.86** | clean (peak) |
| 18 | 6 | 0.99 | +0.03 | washed out |
| 24 | 8 | 1.00 | −0.06 | null |
| 30 | 10 | 1.00 | **−0.64** | **reversed** |

This reproduces the historical "effect direction reversal" — confirming it is an
**apparatus/measurement artifact of the proximity-modulated DV at saturation**,
not a scientific finding.

## DV comparison — the displacement DVs are confounded; the proximity DV is not

Running all four candidate DVs on the SAME data (via the extended
`analyze_p7h_real_study.py`) shows the confound is in the **DV family**, not the
sequence length. Event intensity is proximity-modulated, so the aligned
(experiment) arm throttles its own force once it nears the critical point, while
the control arm — staying farther away — keeps receiving full-size nudges. Any
displacement-MAGNITUDE DV therefore favours control over a sequence.

| DV | best regime (12 act) | reversed regime (30 act) | robust? |
|----|----------------------|--------------------------|---------|
| net displacement `total_displacement` | d=+0.82 ✅ | d=−0.64 ✗ | no (only pre-saturation) |
| path length `path_displacement` Σ‖Δ‖ | d=−2.40 ✗ | d=−4.90 ✗ | **no — worse** (control gets bigger steps throughout) |
| **max proximity** | **d=+4.44 ✅ (p=4e-17)** | **d=+3.78 ✅ (p=3e-15)** | **YES** |
| n_critical_crossings | exp 1.0 / ctrl 0.0 (p≈0)* | d=+2.82 ✅ | yes (count; *degenerate Cohen's d when exp var=0) |

**Path length is NOT the fix** (it amplifies the artifact). The unconfounded DV
is **proximity-based**: it rewards reaching the target the manipulation actually
aims at (the bifurcation), independent of how the modulation scales force.

### Recommendation for the real-human study

1. **Switch the primary objective DV to `max_proximity`** (continuous,
   unconfounded, saturation-robust: d≈+3.8 to +4.4 across all regimes). This
   also revives the *originally* pre-registered crossing-based DV
   (`n_critical_crossings`), which was abandoned only because the OLD broken
   apparatus saturated everything to proximity=1.0 — no longer true after the
   Space A/B fix. Prefer `max_proximity` over the count to avoid the zero-variance
   degeneracy.
2. If net displacement is kept as a secondary DV, **cap the event sequence so the
   experiment arm stays sub-saturation** (max proximity < ~0.95 ≈ ≤4 events at
   `intensity_scale=1.0`); it is only valid pre-saturation.
3. With ΔP feedback ON, common-mode personality drift adds variance that further
   dilutes displacement DVs; the proximity DV is unaffected.

## Reproduce

```bash
./venv/bin/python scripts/experiments/run_p7h_engine_sim.py \
    --n-players 60 --agents 50 --n-actions 12 --event-every 3 --no-feedback \
    --out reports/experiments/p7h_engine_sim
./venv/bin/python scripts/experiments/analyze_p7h_real_study.py \
    --sessions reports/experiments/p7h_engine_sim/p7h_player_test_sessions.json \
    --survey   reports/experiments/p7h_engine_sim/p7h_survey_responses.json \
    --out      reports/experiments/p7h_engine_sim
```

> NOTE: the survey numbers (H2) in this run are **synthetic** (parametric model
> in `synthesise_survey`) and only validate the analysis plumbing — they are not
> evidence about subjective UX.
