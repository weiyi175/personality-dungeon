# B1 Smoke Lock Sheet

## 1. Scope

- Protocol: `simulation.b1_async_dispatch_gate`
- Stage: smoke only
- No large sweep, no gate60, no confirm block expansion
- Working hypothesis family: event coupling + world-state coupling
- Explicitly excluded families: subgroup coupling, leader policy coupling

## 2. Core Hypothesis

The active mechanism under test is not just event dispatch noise. The hypothesis is that event timing plus world-state feedback creates a non-isomorphic coupling channel that changes the system in a way that cannot be reduced to:

- time-varying weights only
- simple parameter drift from the baseline
- a baseline-equivalent non-isomorphic variant

If the observed change can be reproduced by those weaker forms, this proposal fails the irreducibility test.

## 3. Success / Failure Definition

### 3.1 Smoke success

- `fairness_fail_count = 0`
- all smoke artifacts are produced successfully
- no protocol invariant is violated
- no unsupported coupling family is enabled

### 3.2 Smoke failure

- any fairness failure in smoke
- missing or malformed `events_json`
- any world-feedback mode conflict
- any attempt to mix in subgroup or leader-policy mechanisms
- any output path or seed plan mismatch

### 3.3 Deferred gate thresholds

These are locked for later gate work, but not used as the smoke pass/fail rule in this phase:

- `L1 <= 3`
- `Healthy >= 42`
- `new_L1 = 0`

## 4. Main Indicators

### 4.1 Primary indicator for this smoke phase

- `fairness_fail_count`

### 4.2 Secondary indicators to monitor

- `mean_stage3_score`
- `mean_env_gamma`
- `phase direction consistency`
- `entropy`
- `action_diversity`
- `world-state deviation`
- `boundary_hit_rate`

### 4.3 Interpretation rule

- If `entropy` drops but Level 3 does not appear later, mark it as a local signal only.
- If `action_diversity` rises but rotation does not form, mark it as a local signal only.
- Do not promote secondary indicators to the main gate in this smoke phase.

## 5. Fixed Parameters

| Parameter | Value | Status |
| --- | --- | --- |
| Python | `./venv/bin/python` | fixed |
| Python version | `3.10.12` | fixed |
| `n_players` | `300` | fixed |
| `n_rounds` | `12000` | fixed |
| `events_json` | `docs/personality_dungeon_v1/02_event_templates_smoke_v1.json` | fixed |
| `smoke_seeds` | `42,44,45,67,73,90` | fixed |
| `event_dispatch_fairness_window` | `2000` | fixed |
| `event_dispatch_fairness_tolerance` | `0.50` | fixed |
| `event_dispatch_target_rate` | `0.08` | fixed |
| `smoke_out_json` | `outputs/b1_async_dispatch_smoke_summary.json` or run-specific smoke path | fixed |
| `gate_out_json` | not used in smoke-only phase | deferred |

## 6. Scan Parameters

Only the following are allowed to vary in this smoke-only lock sheet:

| Parameter | Allowed values | Notes |
| --- | --- | --- |
| `event_dispatch_mode` | `async_round_robin`, `async_poisson` | compare exactly these two modes only |
| `world_feedback_mode` | `off`, `adaptive_world`, `read_only`, `difficulty_only` | use one mode per run; do not mix modes |
| `replicator_update_mode` | protocol default or explicitly documented async control | keep fixed inside a cell |
| `event_impact_mode` | protocol default only | do not sweep in smoke |
| `event_reward_mode` | protocol default only | do not sweep in smoke |

Any additional scan requires a separate spec.

## 7. Parameter Lock Table

| Parameter | Status | Rule |
| --- | --- | --- |
| `players` | fixed | `300` |
| `rounds` | fixed | `12000` |
| `seeds` | fixed | smoke seed set only |
| `memory_kernel` | missing / not in scope | do not import from H-series |
| `selection_strength` | missing / not in scope | do not import from H-series |
| `init_bias` | missing / not in scope | do not import from H-series |
| `a` | missing / not in scope | do not import from matrix bridge |
| `b` | missing / not in scope | do not import from matrix bridge |
| `matrix_cross_coupling` | missing / not in scope | do not import from matrix bridge |
| `series` | missing / not in scope | not part of B1 smoke |
| `eta` | missing / not in scope | not part of B1 smoke |
| `stage3_method` | missing / not in scope | not part of B1 smoke |
| `phase_smoothing` | missing / not in scope | not part of B1 smoke |
| `events_json` | fixed | smoke template only |
| `world_update_interval` | fixed | keep at protocol default unless the protocol explicitly says otherwise |
| `state_ranges` | missing | not exposed in the current B1 CLI surface |
| `clamp` | missing | enforced in the runtime / template, not via this smoke sheet |

## 8. Coupling Rules

### 8.1 Allowed coupling family for this run

- event dispatch coupling
- event impact coupling
- world-state feedback coupling

### 8.2 Forbidden families for this run

- subgroup coupling
- leader policy coupling
- H2 `threshold_ab`
- H3 fixed-subgroup / hybrid families
- any leader-policy bridge that rewrites the payoff operator directly

### 8.3 Mutual exclusions

- `world_feedback_mode = off` cannot be treated as the active coupling condition in the same run where `adaptive_world` is the tested mechanism.
- `world_feedback_mode = read_only` is control-only and must not be counted as active world coupling.
- `event_dispatch_mode` must be a single mode per cell; do not combine `async_round_robin` and `async_poisson` in one run.
- `event_reward_mode` values are mutually exclusive.
- `event_impact_mode` values are mutually exclusive.
- `event_modulation_mode` values are mutually exclusive.
- `require_async_update` must not be mixed with a sync-only interpretation.

## 9. Irreducibility Check

This proposal is only valid if the observed signal requires the coupled event/world mechanism itself.

Fail the proposal if the same result can be reduced to either of the following:

- a pure baseline with time-varying weights
- a baseline plus parameter drift with no genuine coupling topology change

Pass only if the coupling topology contributes a new structure that the baseline cannot reproduce.

## 10. Smoke Execution Rule

1. Run smoke only.
2. Do not launch a large sweep.
3. Do not open a gate60 block unless the smoke phase itself is explicitly promoted later.
4. Keep all outputs in the standard `outputs/` naming convention.
5. Record provenance for every smoke artifact.
6. Monitor world-state deviation extremes: if any world-state variable reaches a runtime hard-coded clamp boundary (e.g., exactly `0.0` or `1.0`), mark the run as `Risk: Boundary Hit` and fail the smoke run for investigation.

## 11. Missing Items

- `state_ranges`: missing
- `clamp`: missing at CLI level, runtime-enforced only
- `memory_kernel`, `selection_strength`, `init_bias`, `a`, `b`, `matrix_cross_coupling`, `series`, `eta`, `stage3_method`, `phase_smoothing`: missing for this B1 smoke sheet because they belong to other families or other runtime paths

- Entropy base (verification): `weight_entropy` and other entropy indicators must use natural log (ln). I verified `evolution/independent_rl.py::weight_entropy` uses `math.log` (natural log). If any other entropy implementation uses `log10`, `log2`, or explicit base-arguments to `math.log(..., base)`, replace with `math.log` or `np.log` and update any thresholds documented in the spec accordingly.

## 12. Data Contract Validation (CI)

To prevent mixed JSON formats from breaking gate logic, schema validation must be routed by payload identity.

### 12.1 Payload contracts

- `multi_run_summary`: strict report contract for aggregated multi-run outputs.
- `legacy_summary`: legacy smoke/gate summary contract with keys such as `total_seeds`, `l1`, `l2`, `l3`, `outcomes`.
- `provenance`: runtime provenance contract; allows extra debug fields but enforces core physical checks.

### 12.2 Validation behavior

- `multi_run_summary`: strict (`extra=forbid`).
- `legacy_summary`: structure and consistency checks must pass:
	- `total_seeds == len(seeds) == len(outcomes)`
	- `l1 + l2 + l3 == total_seeds`
	- `outcomes.seed` set equals `seeds` set
- `provenance`: tolerant (`extra=ignore`) for forward compatibility, but physical consistency is required:
	- if present, `difficulty_index_mean` and `mean_difficulty_index` must satisfy `0 <= d <= 1`.

### 12.3 CI split-gate command

Use type-specific gate checks to avoid cross-pipeline false failures:

1. `./venv/bin/python check_schema.py outputs --recursive --only-type provenance`
2. `./venv/bin/python check_schema.py outputs --recursive --only-type legacy_summary`
3. `./venv/bin/python check_schema.py outputs --recursive --only-type multi_run_summary`

### 12.4 Health report requirement

Every schema run must emit a Stage 3 health report with:

- `ok / invalid / skip` counts by payload type
- `filtered_out` count when `--only-type` is used
- non-zero exit code on validation failure

## 13. H3 Handoff Rule (Delayed Feedback Branch)

When moving to Hypothesis 3 (delayed or lagged world feedback), this lock sheet remains the control contract and must not be silently reinterpreted.

### 13.1 Scope split (mandatory)

- H3 runs are a separate branch protocol; do not merge H3 outputs into B1 smoke pass/fail tables.
- B1 smoke artifacts remain the reference control for H3 comparisons.
- Any H3 parameter (delay/smoothing/history) must be declared in a separate spec block in `SDD.md` before runtime changes.

### 13.2 Invariants carried into H3 smoke

- Keep using `./venv/bin/python` and Python `3.10.12`.
- Keep fixed seeds `42,44,45,67,73,90` for the first H3 scout.
- Keep `events_json`, fairness window/tolerance, and target rate exactly as this lock sheet.
- Keep stage as smoke only; no gate60 expansion until H3 smoke is explicitly promoted.

### 13.3 Output and provenance requirements

- Use a dedicated output namespace (for example `outputs/personality_rl_async_poisson_h3_delay_*`).
- Every H3 artifact must include provenance fields for the added delay/smoothing controls.
- If provenance is missing H3 controls, mark artifact invalid for H3 interpretation.
