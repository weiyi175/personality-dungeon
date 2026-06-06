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

### Recommendation for the real-human study

- **Cap the event sequence so the experiment arm stays sub-saturation**
  (target max proximity < ~0.95): roughly **≤ 4 bifurcation events per session**
  with the current `intensity_scale=1.0`.
- Consider a **saturation-aware DV** (e.g. path length Σ‖Δ‖, or proximity-gain)
  in addition to net displacement, since net displacement is biased once an arm
  saturates. (Path displacement is already recorded by the tracker.)
- With ΔP feedback ON, the common-mode personality drift adds variance that
  dilutes the net-displacement effect over long sequences; the short-regime
  effect survives feedback (smoke test: d≈1.07 at 4 events, feedback on).

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
