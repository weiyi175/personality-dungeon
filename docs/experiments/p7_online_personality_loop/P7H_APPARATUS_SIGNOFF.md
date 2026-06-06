# P7-H Apparatus Readiness — Sign-off & Godot Integration Checklist

**Date:** 2026-06-06   **Branch:** `feature/phase4-todos` (7 commits, **not pushed**)
**Purpose:** one-page status + the exact steps to wire the Godot frontend when
back at the screen, before real-human collection.

---

## 1. Status at a glance

| Item | State | Evidence |
|------|-------|----------|
| Backend apparatus bugs (500s, non-reproducible hash) | ✅ fixed | `79e43bf` |
| Space A ↔ Space B coordinate mismatch | ✅ fixed (bijection) | `8508266`, `simulation/personality_space.py` |
| Engine event path (persists across rounds) | ✅ added | `8508266`, `RLSessionEngine.apply_personality_event` |
| Primary objective DV decided | ✅ `max_proximity` | `fa3c24f`, pre-reg §4 |
| Operating regime (saturation reversal) understood | ✅ mapped + capped | `382a16a`/`80e1916`, REGIME_FINDING.md |
| Robustness (seed/intensity/feedback) | ✅ d≥1.84, 100% sig | `d8f2f82`, robustness_sweep.json |
| Regression test locking behaviour | ✅ | `9a4a42d`, `tests/test_p7h_apparatus_regression.py` |
| **Godot frontend wiring + in-engine smoke test** | ⛔ **PENDING (needs screen)** | this doc §4 |
| Real-human collection | ⛔ blocked on Godot | — |

**Bottom line:** everything verifiable in pytest is done and locked. The only
remaining work before collection is connecting the frontend to the validated
backend (§4) and one in-engine smoke test (§5).

---

## 2. What was wrong, and the fix (one paragraph)

Events were designed in **Space A** (the RL-engine/SVD coordinates where
`BASELINE_ATTRACTOR`, `v1`, `v2`, `ε_c` live) but the frontend applied and
measured them in its own **Space B** (an ad-hoc mapping from population action
proportions `p_aggressive…`), on a different numeric scale. Result: proximity
saturated instantly and event nudges did not persist or register in the DV — the
historical "effect reversal." Fix: a canonical, exactly-invertible **A↔B affine
bijection** (`personality_space.a_to_b/b_to_a`, recentre on the baseline
attractor, scale by `ε_c_app≈0.11`) plus an **engine-side Space-A event path** so
events are designed, applied, and measured in the same space.

---

## 3. DV & regime decisions (locked in pre-registration §4/§6)

- **Primary objective DV = `max_proximity`** (Welch one-sided, exp>ctrl).
  Unconfounded by the proximity-modulated event intensity. Validated d=4.44
  (best regime); robust d=2.44–5.26 across feedback/intensity/seed, 100%
  significant.
- **Net displacement = secondary (H1b)**, valid only sub-saturation.
- **Do NOT use path length** as a DV — it amplifies the artifact (proven: d=−2.4
  to −4.9).
- **Event-sequence cap:** keep the experiment arm's max proximity < ~0.95 →
  roughly **≤ 4 bifurcation events per session** at `intensity_scale=1.0`
  (replaces the old "30 step" design expectation; 30 steps ≈ 10 events lands in
  the reversed regime).

---

## 4. Godot integration checklist (do this at the screen)

The backend is now authoritative for the 9D personality (Space A). The frontend
should stop holding its own Space-B personality and instead drive the engine.

- [ ] **Init with events enabled.**
  `POST /rl_sessions/initialize` with `{"space_a_events_enabled": true, ...}` →
  `session_id`. (Default is `false`, which leaves legacy sessions unchanged.)

- [ ] **Apply each bifurcation event through the backend** instead of mutating a
  local personality:
  `POST /rl_sessions/{id}/apply-event`
  - designed event: `{"group": "experiment"|"control", "intensity_scale": 1.0, "seed": <int>}`
    (experiment = v1-aligned, control = random direction of **equal magnitude**),
  - or explicit: `{"displacement": [9 floats]}` (Space A).
  Response is a full snapshot incl. `mean_personality` (Space A) and
  `personality_displacement` (DV).

- [ ] **Read state for display / DV** via
  `GET /rl_sessions/{id}/personality` →
  `mean_personality_space_a`, `mean_personality_space_b`, `personality_displacement`,
  `bifurcation` (proximity). Use `space_b` for display if you keep a normalised
  view; it comes from the canonical bijection.

- [ ] **Reconcile / remove the old Space-B mapping.** Replace the frontend's
  `p_aggressive→personality` mapping with `personality_space.a_to_b()` (or just
  read `mean_personality_space_b` from the endpoint). Trait names/order already
  match; only the scale differed.

- [ ] **Stop the per-round overwrite.** `PlayableLoopController.gd:256` calls
  `set_personality_manual()` every RL round, which wipes event perturbations.
  Remove it; let the engine own personality and let events accumulate via
  `apply-event`.

- [ ] **Cap events per session ≤ 4** (see §3) in the game/event scheduler.

---

## 5. In-engine smoke test (after §4)

- [ ] Run one experiment and one control session through the real game loop.
- [ ] Confirm: (a) control proximity does **not** sit at ~1.0 from the start
  (Space A/B fix holds), (b) `personality_displacement` grows after an
  `apply-event` and **persists** across subsequent rounds, (c) experiment reaches
  higher `max_proximity` than control.
- [ ] Spot-check that `GET .../personality` values match what the UI shows.

---

## 6. Reproduce the backend validation

```bash
# Apparatus validation (engine Space-A path) + pre-registered analysis
./venv/bin/python scripts/experiments/run_p7h_engine_sim.py \
    --n-players 60 --agents 50 --n-actions 12 --event-every 3 --no-feedback \
    --out reports/experiments/p7h_engine_sim
./venv/bin/python scripts/experiments/analyze_p7h_real_study.py \
    --sessions reports/experiments/p7h_engine_sim/p7h_player_test_sessions.json \
    --survey   reports/experiments/p7h_engine_sim/p7h_survey_responses.json \
    --out      reports/experiments/p7h_engine_sim

# Primary-DV robustness sweep
./venv/bin/python scripts/experiments/sweep_p7h_robustness.py \
    --seeds 8 --sessions 40 --out reports/experiments/p7h_engine_sim

# Regression guard
./venv/bin/python -m pytest tests/test_p7h_apparatus_regression.py -q
```

Full write-up: [reports/experiments/p7h_engine_sim/REGIME_FINDING.md](../../../reports/experiments/p7h_engine_sim/REGIME_FINDING.md).
Pre-registration: [P7H_PREREGISTRATION.md](P7H_PREREGISTRATION.md).

---

## 7. Sign-off

| Role | Item | Signed |
|------|------|--------|
| Backend apparatus | validated in simulation, regression-locked | ✅ (this work) |
| DV / pre-registration | max_proximity primary, regime capped | ✅ (pre-reg updated) |
| Frontend wiring (§4) | — | ☐ pending (at screen) |
| In-engine smoke (§5) | — | ☐ pending |
| Go for real-human collection | — | ☐ after §4–§5 |
