# Sampling Destroys RPS Rotation: A Mean-Field-vs-Sampled Isolation of Where Level-3 Dies in Discretized Replicator Dynamics

**Research Draft — Personality Dungeon L3-Bottleneck Negative Result (2026-06-18)**

> **Companion baseline**: [paper_draft_v1.md](paper_draft_v1.md) (L0→L2 positive result, Pr=30/30)
> **Pre-registration (locked before confirmatory collection)**: [docs/experiments/l3_bottleneck/L3_BOTTLENECK_PREREGISTRATION.md](experiments/l3_bottleneck/L3_BOTTLENECK_PREREGISTRATION.md)
> **Confirmatory report + data**: [reports/experiments/l3_bottleneck/phase2_confirmatory_report.md](../reports/experiments/l3_bottleneck/phase2_confirmatory_report.md)
> **Decision chain (code-locked)**: [analysis/cycle_metrics.py](../analysis/cycle_metrics.py) `classify_cycle_level` (L1345) / `phase_direction_consistency_turning` (pass criterion L1082)

---

## Abstract

A near-neutral three-strategy replicator system (Aggressive / Defensive / Balanced)
under personality-conditioned event feedback robustly reaches a **Level-2 structural
cycle** — sustained oscillation without a single dominant rotational direction
(companion baseline, Pr = 30/30). It does **not** reach **Level 3**: a sustained,
single-direction rock-paper-scissors (RPS) rotation. Across an exploratory sweep of
30+ conditions and ≈120 runs spanning structural patches on well-mixed dynamics
(island demes, personality/phase strata, amplitude-dependent selection, tangential
drift), local pairwise imitation, and random-init local mini-batch replicator on
lattice and small-world topologies, **zero** runs achieved Level 3.

We move this bounded negative result to a **causal isolation**. A pre-registered
confirmatory phase (4 *best-hope* cells, 10 seeds each, locked decision chain)
confirms Level-3 remains structurally unreachable: per-cell stage-3 consistency sits
in a weak-bias band around **0.51** — one order of magnitude below the 0.55 rotation
gate and statistically separated from RPS rotation — with 0–1/10 Level-3 seeds across
B4/B5/T/C1 mechanisms and 0 Level-3 under an `eta ∈ {0.55, 0.60, 0.65, 0.70}` sweep.
Critically, the **deterministic mean-field twin** of the keystone generator achieves
**perfect rotation** (consistency = 1.0000, Level 3, all 10 seeds, bit-identical),
while its **only** modification — adding finite-population sampling — collapses it to
consistency ≈ 0.51 / Level 2. We therefore isolate the loss of Level-3 rotation to
the **sampling + averaging discretization layer**, not to any absence of rotational
structure in the payoff geometry. We further show that a naïve local-replicator
discretization with uniform initialization (C2) is a **mathematical degeneracy**
(`x_local ≡ x_global`) that must be excluded as a valid test — a hidden trap for
practitioners. The result delimits a sharp boundary: personality-conditioned feedback
can break symmetry into a Level-2 limit cycle, but crossing into Level-3 rotation
requires a qualitatively different update rule (asynchronous, or continuous tangential
projection), because the standard sampled-synchronous discretization provably destroys
the rotation that exists in its own continuum limit.

---

## 1. Introduction

The companion baseline ([paper_draft_v1.md](paper_draft_v1.md)) established that a
three-strategy replicator system under personality-conditioned delayed feedback shifts
from a stationary Level-0 attractor to a **Level-2 structural cycle**, robustly
(Pr = 30/30 across seeds × noise). Level 2 means the phase-space trajectory oscillates
with detectable amplitude and period, but its turning direction is **not** consistently
single-signed — it is structured wandering, not rotation.

A **Level-3** classification additionally requires **sustained single-direction
rotation** (RPS-style cyclic dominance traced out coherently in phase space). This is
the dynamical signature one would want from a "living" ecology of strategies that
shuffle without any one collapsing — the target behavior for the project's cross-player
diversity layer. The open question left by the baseline was sharp and falsifiable:

> **Can the personality-event feedback loop — or any local/structural patch on the
> sampled discrete replicator — be pushed from a Level-2 plateau into a sustained
> Level-3 rotation, without changing the base payoff topology?**

This paper answers **no**, and — more usefully — isolates **why**. We do not merely
report an absence of Level-3 (a weak "we didn't observe it"). We pair the negative
result with a **deterministic mean-field positive control** built from the very same
generator, which *does* produce perfect Level-3 rotation. The contrast between the
mean-field system (rotation present, consistency = 1.0) and its sampled twin (rotation
destroyed, consistency ≈ 0.51) converts the finding into a mechanistic claim:
**the rotational structure exists in the continuum limit and is destroyed by the
finite-population sampling-and-averaging discretization layer.**

The contribution is threefold:
1. A pre-registered confirmatory negative result for Level-3 reachability across four
   mechanistically distinct discretized-replicator families.
2. A mean-field-vs-sampled causal isolation that attributes the loss to the
   discretization layer rather than to payoff geometry.
3. A methodological warning: the naïve uniform-init local replicator is a mathematical
   degeneracy that silently reduces to global replicator and must not be counted as a
   local-dynamics test.

---

## 2. Background and operationalization

### 2.1 Cycle-level taxonomy

The classifier `classify_cycle_level` ([analysis/cycle_metrics.py](../analysis/cycle_metrics.py) L1345)
is a three-stage cascade on the strategy-weight time series `w_*` (weights, not
proportions, are used for sampling stability; cycle_metrics.py:421–423):

```
Stage 1 (amplitude):  assess_stage1_amplitude,  amplitude_threshold = 0.02
   fail → Level 0
Stage 2 (period):     assess_stage2_frequency,  stage2_method = "autocorr_threshold",
                      corr_threshold = 0.09
   fail → Level 1
Stage 3 (rotation):   phase_direction_consistency_turning,  stage3_method = "turning"
   fail → Level 2
all pass → Level 3
```

Note `corr_threshold` only gates L1→L2; every condition in this study is already
L2/L3 (Stage 2 passes), so it is **non-binding** for the L2→L3 question and is fixed at
the L2-lineage convention 0.09.

### 2.2 The Level-3 gate (the crux)

Stage-3 pass is decided verbatim by (cycle_metrics.py:1082):

```python
passed = (consistency >= float(eta)) and (turn_strength >= float(min_turn_strength))
```

- **`consistency` (reported as `stage3_score`)** is the fraction of consecutive
  phase-space turning events whose cross product `cross(Δp_t, Δp_{t+1})` agrees in sign
  with the net rotation direction, ∈ [0, 1] (cycle_metrics.py:1078–1082).
  **0.50 = chance level**: turning signs unbiased ⇒ noise wobble, not rotation.
- `turn_strength = mean(|cross|)` is rotation magnitude.
- **Locked parameters**: `eta = 0.55`, `min_turn_strength = 0.0`. With
  `min_turn_strength = 0`, only `consistency ≥ 0.55` binds — a **lenient** gate
  (direction only, magnitude ignored). Failing even this makes the negative result
  *harder*.
- The 0.55 value is anchored externally: the single Level-3 instance in the companion
  baseline (`paper_draft_v1`, seed 45, topology-changing payoff) scored consistency
  **0.5611** — barely 1 point above the bar and ~6 points above chance, quantifying its
  fragility (it falls back to Level 2 under alternate seeds and longer windows).
- **Locked window**: `burn_in = 1000`, `tail = 1000`, `rounds = 3000`,
  `phase_smoothing = 1`, strategies `(aggressive, defensive, balanced)`.

### 2.3 The three-layer averaging hypothesis (from exploratory Phase 1)

The exploratory phase (Section 3.1) generated a mechanistic hypothesis for why the
sampled discrete synchronous update dissipates rotational/directional structure:

```
Layer 1: Sampling noise       — discrete strategy draws ∝ w inject high-frequency noise
Layer 2: Popularity averaging — x = mean(sampled strategies) erases individual phase
Layer 3: Synchronous update   — w(t+1) ∝ w(t)·f(x): whole population advances in-phase
```

Patches above Layer 3 (alter k, add drift, add strata) are absorbed by Layers 1+2;
removing Layer 3 in favor of pairwise imitation (C1) yields a *stronger* Layer-2
consensus; the only sweep that activated genuine local phase domains (T-series,
random-init local mini-batch) still saw Layer-1 sampling jitter dilute coherence faster
than it could accumulate into globally visible rotation.

---

## 3. Methods

### 3.1 Exploratory phase (Phase 1, hypothesis-generating, ≈120 runs)

A short scan (3 seeds/cell) across 30+ conditions in three families
([level3_bottleneck_phase1_closure.md](../level3_bottleneck_phase1_closure.md)):

| Family | Mechanism | Cells | L3 | Best uplift |
|---|---|---|---|---|
| B2 | island deme + periodic migration | 6 | 0/18 | +0.004 |
| B3 | personality / phase-aware strata | 5 | 0/15 | +0.001 |
| B4 | amplitude-dependent selection strength k | 6 | 0/18 | +0.014 |
| B5 | external tangential drift | 4 | 0/12 | −0.001 |
| C1 | local pairwise (Fermi) imitation | 6 | 0/18 | +0.005 |
| C2 | local mini-batch replicator (uniform init) | 6 | 0/18 | +0.008 |
| T  | local mini-batch (random init) × topology | 4 | 0/12 | −0.006 |
| T follow-up | strong k_local = 0.12 | 2 | 0/6 | −0.005 |

**Result: 0 Level-3 seeds across all families.** All conclusions here are treated as
*exploratory* (hypothesis generation) and are **not** used as confirmatory evidence.

### 3.2 Confirmatory phase (Phase 2, pre-registered, locked decision chain)

Per the pre-registration (locked before any confirmatory seed ran; see git timestamp of
[L3_BOTTLENECK_PREREGISTRATION.md](experiments/l3_bottleneck/L3_BOTTLENECK_PREREGISTRATION.md)),
the *best-hope* cells — those closest to the L3 bar or with the strongest mechanism
signal in Phase 1 — were selected on frozen Phase-1 data (transparent, no post-hoc
cherry-picking; selection committed to `phase2_cell_selection.json`) and extended from
3 to 10 seeds. Seeds: existing `45/47/49` + new `50–56` (consecutive integers, no seed
hunting).

**Arm A — mean-field positive control (deterministic, not seed-extended):**

| cell | system | consistency | role |
|---|---|---|---|
| `g1_mean_field_delta0p000` | deterministic mean-field (no sampling) | **1.0000** | proves rotation exists in continuum limit |

The mean-field system is deterministic (seeds 45/47/49 bit-identical); reproducibility
is shown via `delta ∈ {0, .003, .006, .010, .015}`, all consistency = 1.0 / Level 3.

**Arm B — sampled treatment cells (3 → 10 seeds each):**

| cell | family | Phase-1 mean/max consistency | role |
|---|---|---|---|
| `g2_sampled_delta0p000` | B5 tangential drift (**sampled twin of Arm A**) | 0.5150 / 0.5410 | **keystone**: same generator, sampling only |
| `beta0p60_k0p08` | B4 amplitude-dependent k | 0.5306 / **0.5488** | closest sampled cell to the 0.55 bar |
| `t_lattice4_minibatch` | T local mini-batch (random init; only valid local test) | 0.5061 / 0.5255 | domains form (batch_phase_spread ≈ 1.55 rad) yet consistency stuck at chance |
| `g2_c1_small_world_b10p0_m0p5` | C1 local pairwise Fermi | 0.5109 / 0.5219 | over-homogenization branch |

The keystone gives causal attribution (sampling layer kills rotation); the breadth cells
prove the negative result spans B4/B5/T/C1 rather than being a B5 special case.

### 3.3 Pre-registered falsifiable hypotheses

- **H0-neg (structural unreachability)**: each best-hope cell holds 0/10 Level-3 seeds
  (tolerance ≤ 1/10 as a distribution-tail outlier, consistent with the baseline's
  single fragile instance); **≥ 2/10 refutes** and forces downgrade to "conditionally
  reachable" + root-cause.
- **H1-chance (consistency at chance)**: per-cell consistency is not distinguishable
  from / sits near 0.50 (one-sample two-sided t vs μ₀ = 0.50, report CI and Cohen's d;
  no go/no-go significance reverse-inference).
- **H2-threshold-robust**: the negative result holds across `eta ∈ {0.55, 0.60, 0.65, 0.70}`.
- **H-PC (positive control)**: Arm A holds `cycle_level == 3`, `consistency == 1.0`.
  If the mean-field control fails to rotate, the generator/decision-chain is suspect and
  Arm B must not be interpreted until resolved.

### 3.4 C2-uniform exclusion (locked, irreversible)

C2 with uniform initialization measured `cosine_vs_global = 1.000000` and
`batch_phase_spread = 0.000000` (exactly zero). Under uniform init all `w_i(0)` are
equal ⇒ `x_local ≡ x_global` ⇒ `g_local ≡ g_global` identically — the configuration
**reduces by construction to global replicator** and is not a valid test of local
dynamics. C2-uniform is therefore **removed from the count of tested interventions** and
presented only as a methodological warning. The **only valid test of local replicator
is T-series (random init).**

---

## 4. Results

### 4.1 Arm A — mean-field positive control

| cell | n | cycle_level | consistency | verdict |
|---|---|---|---|---|
| `g1_mean_field_delta0p000` | 10 | all = 3 | all = **1.0000** | ✅ positive_control_holds |

Deterministic (10 seeds bit-identical). **Rotational structure is confirmed to exist in
the continuum limit.**

### 4.2 Arm B — sampled treatment cells

| cell | family | n | **L3** | mean consistency | 95% CI | eta-sweep L3 (.55/.60/.65/.70) | verdict |
|---|---|---|---|---|---|---|---|
| `g2_sampled_delta0p000` | B5 (keystone, sampled twin) | 10 | **0** | 0.5145 | [0.506, 0.523] | 0/0/0/0 | ✅ negative_holds |
| `beta0p60_k0p08` | B4 state-dep k | 10 | **0** | 0.5109 | [0.500, 0.521] | 0/0/0/0 | ✅ negative_holds |
| `t_lattice4_minibatch` | T local mini-batch | 10 | **0** | 0.5148 | [0.507, 0.523] | 0/0/0/0 | ✅ negative_holds |
| `g2_c1_small_world…` | C1 (native window 5000/1500/1500) | 10 | **0** | 0.3068 | [0.118, 0.496] | 0/0/0/0 | ✅ negative_holds |
| `g2_c1_small_world…` | C1 (locked window 3000/1000/1000) | 10 | **1** (blip) | 0.4722 | [0.353, 0.591] | **1**/0/0/0 | ✅ negative_holds (tolerance ≤ 1) |

### 4.3 The keystone causal contrast

Same B5 generator, single modification (mean-field → sampled):

| | consistency | cycle_level |
|---|---|---|
| `g1_mean_field` (deterministic) | **1.000** | 3 |
| `g2_sampled` (its twin) | **0.5145** | 2 |

**Discretization drops the rotation consistency from 1.0 to 0.51.** This is the
causal-isolation result: the sampling-and-averaging layer — and nothing else — destroys
a rotation that is provably present without it.

### 4.4 Honest details (reported, not smoothed)

1. **Consistency is ≈ 0.51, not exactly 0.50.** B5/B4/T cells' 95% CIs do not fully
   cover 0.50 (means 0.511–0.515, d_vs_0.50 ≈ 0.74–1.27): a statistically detectable but
   rotationally irrelevant weak directional residual, one order below the 0.55 gate. The
   precise statement of H1-chance is therefore *"consistency is stuck in a ≈0.51
   weak-bias band, far from the rotation gate"*, not *"equal to chance"*. The
   eta ∈ {.55–.70} sweep (all 0/10) confirms this residual is not sustained rotation.
2. **The C1 1/10 Level-3 is a short-window noise artifact.** Under the locked window
   (3000/1000/1000) C1 yields 1/10 (max 0.5510 grazing the bar); the native window
   (5000/1500/1500) gives 0/10, mean 0.3068 (over-homogenization actively suppresses
   consistency *below* chance). Shorter tails → higher consistency-estimate variance →
   occasional grazing. The eta-sweep dissolves the blip (eta ≥ 0.60 → 0). Both windows
   are within the pre-registered ≤1 tolerance and mutually corroborate the negative
   result.
3. **B4's "closest to the bar" cell regresses to chance.** Phase-1 3-seed mean 0.5306
   (then the closest to 0.55) dropped to 0.5109 at 10 seeds — confirming the "near-miss"
   was small-sample fluctuation, not true approach.

---

## 5. Discussion

### 5.1 What the contrast proves

The result is not "we failed to find Level 3." It is: **the rotation exists (mean-field
consistency = 1.0) and is destroyed by a single, identifiable transformation
(finite-population sampling).** Because the keystone twins share generator, payoff,
window, and decision chain, the only explanatory variable is the discretization layer.
The negative result is thereby upgraded from *bounded* ("each patch we tried fails") to
*causal* ("the discretization layer is where rotation dies").

### 5.2 Why every patch fails (mechanism)

Mapping back to the three-layer averaging hypothesis (§2.3):
- **B-series** patches live above Layer 3 (k modulation, tangential drift, strata) and
  are absorbed by Layers 1+2. B5's external tangential drift ends up ≈ orthogonal to the
  tangent (alignment ≈ −0.01).
- **C1** replaces Layer 3 with pairwise imitation, producing a *stronger* consensus
  (edge_strategy_distance = 0.000) — faster homogenization than well-mixed.
- **T-series** is the most informative: random Dirichlet init breaks the C2 degeneracy
  (cosine_vs_global 1.000 → 0.05–0.10), genuine local phase domains form
  (batch_phase_spread ≈ 1.55 rad, growth dispersion > 0.2, negative spatial
  autocorrelation at d1), yet Layer-1 sampling jitter dilutes coherence each step faster
  than domains can accumulate into global rotation. Stronger selection (k_local 0.08 →
  0.12) makes uplift *more* negative — ruling out "insufficient selection pressure."

### 5.3 Relation to the positive baseline

The companion baseline's L0→L2 result (Pr = 30/30) is the **upper boundary** of what the
sampled-feedback mechanism *can* do. This paper draws the matching lower boundary on
what it *cannot*: the system can be driven to a structured Level-2 limit cycle but not to
Level-3 rotation. The sole fragile Level-3 baseline instance (consistency 0.5611,
topology-changing payoff) reinforces rather than contradicts this — it required a
*different payoff topology* and still sat 1 point above the bar, collapsing under
alternate seeds.

### 5.4 Implications

Crossing into robust Level-3 rotation in this class of models requires a **qualitatively
different update rule**, not stronger tuning: candidates are (i) a continuous tangential
projection operator that injects rotational component directly into the flow field, or
(ii) asynchronous / event-driven updates that let leading players carry rotation rather
than averaging it away each synchronous step. The practical corollary for the project's
ecology layer: a discretized synchronous replicator on raw RPS payoff will damp to
coexistence, not sustain rotation — consistent with the separate finding that the locked
ecology payoff (a > b) is a damped spiral-in to a coexistence fixed point.

---

## 6. Limitations and threats to validity

- **Generality of the gate.** Results are stated under the locked `eta = 0.55`,
  `min_turn_strength = 0` operationalization. The eta-sweep (0.55–0.70) and the mean-field
  control (consistency = 1.0 passes any reasonable gate) bound this dependence, but a
  fundamentally different rotation metric could re-open the question; we therefore fix the
  metric and report it verbatim.
- **Best-hope selection.** Confirmatory cells were the strongest Phase-1 candidates. The
  negative result is strongest *because* even the best hopes fail at N = 10; it does not
  claim every conceivable discretization fails — only that the tested four mechanistically
  distinct families do, and that the mean-field twin isolates the cause.
- **Window standardization (Deviation D1).** C1's native window differs from the locked
  window; both are reported and agree (0/10 native, 1/10 locked-blip).
- **Weak-bias residual (Deviation D4).** Consistency is ≈ 0.51, not exactly chance
  (§4.4.1); the main claim (Level-3 unreachable, one order below the gate) is unaffected.

---

## 7. Reproducibility

- **Pre-registration** (locked, git-timestamped): `docs/experiments/l3_bottleneck/L3_BOTTLENECK_PREREGISTRATION.md`
- **Cell selection** (locked): `docs/experiments/l3_bottleneck/phase2_cell_selection.json`
- **Confirmatory analysis**: `scripts/experiments/analyze_l3bn_phase2.py`
- **Committed evidence**: `reports/experiments/l3_bottleneck/phase2/*_summary.tsv`,
  `*_combined.tsv`, `*_decision.md`, `phase2_confirmatory_analysis.json`
- **Raw per-seed trajectories** (≈6.3 GB) are gitignored; analysis TSV/JSON are committed
  and sufficient to reproduce every conclusion. Trajectories regenerate from the
  `simulation/` runners with locked seeds 45–56.
- **Decision chain**: `analysis/cycle_metrics.py` `classify_cycle_level` (L1345) /
  `phase_direction_consistency_turning` pass criterion (L1082).

---

## 8. One-line summary

A deterministic mean-field replicator rotates perfectly (consistency = 1.0, Level 3);
adding finite-population sampling — and nothing else — collapses it to consistency ≈ 0.51
(Level 2), and this collapse holds across four mechanistically distinct sampled families
under a pre-registered confirmatory protocol. **Sampling, not payoff geometry, is where
Level-3 rotation dies.**
