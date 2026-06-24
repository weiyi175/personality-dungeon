# L3-BN Metric-Sensitivity Supplementary — turning-consistency vs winding-number

> **Status (2026-06-22): 🚫 dynamics 線 SHELVED.** 文獻精讀（6 篇）裁決：本研究 dynamics 現象＝**獨立重新發現既有有限族群 quasi-cycle / coherence-resonance**（Traulsen-Claussen-Hauert 2006 的 1/√N；McKane-Newman 2005 PRL；arXiv:1006.0825 van Kampen RPS quasi-cycle；Yang-Rogers-Dawes 2017）。b4 的 state-dependent selection-*intensity* 旋鈕字面不在 constant-w 文獻，但未做 Hopf/bifurcation 分析、未隔離 state-dependence 是否本質 ⇒ 不成獨立貢獻。`paper_draft_v2` / `draft_l3_bottleneck` **封存**（乾淨封存，非失敗）。研究價值改框＝方法論示範 + 跨方法複現 + 指標陷阱案例。**完整文獻對比 → `../literature_positioning_report.md`。** §175 的 rotation 定義決策因此 moot（封存不需定）。
> **Origin:** began as the post-hoc cross-protocol re-run of the `paper_draft_v1` L0→L2 positive control (pre-reg §2 locked it as `不重跑`); the re-run surfaced a series-choice (`p_`/`w_`) and then a **metric-definition** dependence that is now the dominant question.
> **No re-simulation** beyond the positive control: all cell analyses re-classify the surviving Phase-2 per-seed trajectory CSVs in `../phase2/<cell>/seed*.csv`.
> **Data:** `metric_sensitivity_data.json`. **Scripts (reproducible):** `l3pc_rerun.py`, `l3_wcheck2.py`, `l3_gate_sweep2.py`, `l3_phase_rel.py`, `l3_b4_winding_robust.py`, `l3_consolidate.py`.

---

## R1 — Positive control re-run under protocol-2 (`p_` chain): 30/30 Level 2

Seeds 1–10 × σ∈{0.005,0.05,0.20}, rounds=3000, popularity=expected, classified through the
**as-executed** protocol-2 chain (`p_*` proportions; burn_in=1000, tail=1000, eta=0.55,
min_turn_strength=0, turning, phase_smoothing=1 — the exact chain `b5_tangential_drift._seed_metrics` ran).

- **L≥2 = 30/30**, L3 = 0/30, stage3 consistency mean **0.513** [0.487–0.539], same ~0.51 band as the negative cells.
- Cross-protocol hole sealed **on the `p_` chain**. (Data: `../phase2_positive_control_protocol2/`.)

## R2 — Pre-reg `w_*` prose ≠ runner `p_*` code; on `w_` the result flips for 2 of 4 cells

Pre-reg §5 prose names the **`w_*` weight series** as binding; the committed runner classified Arm A,
Arm B, and the positive control all on **`p_*`**. Re-classifying the confirmatory cells on both:

| cell | `p_` L3 / cons | `w_` L3 / cons |
|---|---|---|
| Arm A mean-field (control) | 10/10 · 1.000 | 10/10 · 1.000 |
| b5 g2_sampled (keystone neg) | 0/10 · 0.5145 ✓ (matches committed) | 0/10 · 0.508 |
| **b4 state-k (neg)** | 0/10 · 0.5109 ✓ | **10/10 · 0.711** |
| **t minibatch (neg)** | 0/10 · 0.5148 ✓ | **10/10 · 0.961** |
| c1 small_world (neg) | 1/10 · 0.472 | runner emits no `w_` columns |

`p_` re-classification reproduces the committed values exactly → chain faithful → the `w_` flip is real.

## R3 — Gate sweeps on `w_` (turn_strength yardstick = Arm A genuine rotation = 4.83e-07)

| cell | `w_` consistency | `w_` turn_strength | ×ArmA | dies at min_turn_strength |
|---|---|---|---|---|
| b4 state-k | 0.711 | 1.65e-06 | **3.4×** | 3e-6 (outlasts ArmA, which dies 1e-6) |
| t minibatch | 0.961 | 6.99e-09 | 0.0145× | **3e-8** (collapses immediately) |
| b5 keystone | 0.508 | 3.35e-07 | 0.69× | — (chance direction, never passes eta) |

- eta-sweep (mts=0): t survives to 0.75 (high direction-consistency); b4 to 0.70; b5 never.
- min_turn_strength=0 (the locked gate, chosen to make the *p_* negative *harder*) is what manufactures `w_`-L3.

## R4 — Winding (net signed turns; coherence = |net|/total angular path)

| cell | `w_` net / coher | `p_` net / coher |
|---|---|---|
| Arm A mean-field | −3.25 / 1.000 | −3.25 / 1.000 |
| **b4 state-k** | −2.74 / **1.000** | −2.84 / 0.317 |
| t minibatch | −5.10 / 0.979 (amp 1.5% of genuine) | −5.27 / 0.033 |
| b5 keystone | +4.94 / 0.044 (no coherent winding) | +0.12 / 0.013 |
| c1 small_world | (no `w_`) | −0.07 / 0.023 |

## R5 — b4 `p_` winding robustness → **Case A** (observable rotation a winding classifier flags positive)

| test | b4 `p_` | ArmA (genuine) | b5/t (null) |
|---|---|---|---|
| per-seed net turns (tail 1000) | [−2.7…−3.0], **10/10 same sign**, std ~0.1 | −3.2 ×10 | scattered (b5 4/10, t 7/10) |
| **window scaling** (250/500/1000/2000) | −0.70/−1.39/−2.84/−6.14 **linear** | −0.95/−1.60/−3.25/−6.95 linear | erratic, sign-flips |
| per-quarter (4×250) | [−0.84,−0.76,−0.78,−0.70] **4/4 steady** | 4/4 steady | erratic |
| rate (turns/1000, tail 2000) | **−3.068 ± 0.027** | −3.476 ± 0.000 | b5 −4.77 ± **5.31** |

**b4's observable proportions carry a genuine, steady, constant-rate rotation (~2.85 turns/1000),
quality-indistinguishable from the mean-field control.** The turning-consistency metric reads 0.51
only because sampling jitter inflates the step-local path (coherence 0.32) — it **misses** the rotation.
t's `p_` winding is **not** robust (erratic window-scaling, scattered seeds); b5/c1 have none. **b4 is the sole counterexample.**

---

## R6 — Phase-1 winding re-scan: turning-vs-winding confusion matrix (breadth test)

Both metrics on observable `p_`, all B/C/T families (64 cells, 192 seeds; data survived on disk
in `outputs/*_short_scout/` — no regeneration). Winding+ = per-seed steady rotation (tail-1000 in
4 quarters; 4/4 same sign AND min|quarter net turns| ≥ 0.30; calibrated on anchors ArmA/b4=+, b5/t=−).

```
              Winding+   Winding-
  Turning+        15         0
  Turning-         7       170
```
- **Turning+/Winding− = 0** (no turning false-positives vs winding).
- **Turning−/Winding+ = 7** raw → after filtering **5 genuine**: 2 are **identical C2 files** (lattice==ring;
  the `x_local≡x_global` degeneracy the pre-reg excludes) → removed; 1 is a borderline T seed (last quarter
  at the 0.30 floor). The robust signal = **4 in B4 (state-dependent-k)**, incl. a clean **3/3 cell**
  `beta0p30_k0p08` (window-scaling −3.7 turns/1000, dead steady).

**By family (T+/W+ · T+/W− · T−/W+ · T−/W−):** B2 `0·0·0·21` · B3 `0·0·0·36` · B4 `0·0·4·20` ·
B5 `15·0·0·15` · C1 `0·0·0·42` · C2 `0·0·2·19`(degenerate) · T `0·0·1·17`.

**Verdict: b4 is mechanism-localized — neither singleton nor iceberg.** The "turning systematically
under-detects" scenario is refuted (B2/B3/C1 = zero disagreement). The disagreement is a *reproducible
property of state-dependent selection (B4)*: the proportion orbit rotates steadily but step-locally
low-coherently. The negative result holds for B2/B3/B5-sampled/C1 on both metrics; the *universal* claim
"L3 unreachable in sampled discretized replicator" is false (B4 reaches observable winding-L3 reproducibly).

## R8 — DECISIVE β×cc factorial: state-k is necessary & sufficient; cross-coupling irrelevant (confound resolved 2026-06-21)

**Provenance catch:** every Phase-1/2 cell (all families incl. the B5 negative keystone) ran at
`matrix_cross_coupling = 0.2` (a=1.0) — the topology-changing L3 driver from paper_draft_v1. So the
earlier "b4 escape" was confounded with cc, and the two b4 datasets differ (short-scout near-center vs
Phase-2 near-corner). Resolved by a fresh factorial on the b4 generator (a=1.0, k=0.08, seeds 45/47/49):

```
        cc=0.0 (state-k ALONE)                cc=0.2 (state-k + cc)
β=0.0 | wind+ 0/3  ratio −2.2 (random walk)  | wind+ 0/3  ratio −1.7 (random walk)
β=0.3 | wind+ 3/3  ratio +8.5 (STEADY)       | wind+ 3/3  ratio +9.1 (STEADY)
β=0.6 | wind+ 3/3  ratio +8.2 (STEADY)       | wind+ 3/3  ratio +8.7 (STEADY)
```
**cc=0/β=0.3 → wind+ 3/3, linear scaling (ratio 8.5).** cc=0 ≈ cc=0.2 at every β. ⇒ **state-dependent-k is
necessary AND sufficient for the observable steady winding; cross-coupling is neither necessary nor
contributory** (cc=0.2/β=0 alone = random walk). The "B4 escape = state-k" claim is now clean and
provenance-verified. (turning-L3 = 0/3 everywhere — the metric-disagreement persists, as expected.)

## R9 — B robustness pass: escape is necessary, sufficient, k-flat, cc-independent; a-dependent strength (2026-06-21)

Clean cc=0 b4 runs (state-k alone). Winding criterion as R6.

**B1 — necessity & seed-robustness (10 seeds, k=0.08, cc=0):**
| a | β=0 | β=0.3 |
|---|---|---|
| 1.0 | wind+ 0/10 (random walk) | **wind+ 10/10, ratio +8.8 (steady)** |
| 0.8 | wind+ 0/10 (random walk) | wind+ **7/10**, ratio +3.6 (weaker) |

**B2 — k-threshold (β=0.3 vs β=0 control, cc=0, a=1.0, 3 seeds):** β=0.3 → **3/3 at k=0.06, 0.07, 0.08**
(ratio +8.3/+9.2/+8.5); β=0 → 0–1/3 random-walk. **No k_c ≥ 0.06** — the earlier "k=0.06 fails" was the
cross-coupling confound, not a selection threshold.

**Verdict:** state-dependent-k is **necessary** (β=0 → 0/10 both a) and **sufficient & robust** for the
observable winding escape — **bulletproof at a=1.0 (10/10), k-flat (0.06–0.08), cc-independent** (R8).
**One honest qualification: a-dependence** — strength weakens at a=0.8 (7/10, ratio 3.6). Not a knife-edge;
the *why* (and the a-dependence) remain for the bifurcation analysis (Step C). Data: `b4_B_robustness/`.

## ~~R7 — anti-dominance / lock-in→succession mechanism~~ (RETRACTED 2026-06-21)

> **RETRACTED:** the dominance-level "why" below was **dataset-specific (driven by `a`), not robust.**
> The R8 factorial (a=1.0) shows the *opposite* dominance/argmax pattern from the short-scout (a=0.8):
> β=0 has *more* argmax-flicker (208) and *lower* meanD (0.53); β=0.3 has *fewer* (19) and *higher* meanD
> (0.80). What survives is only the input→output fact (R8): state-k turns a random walk into a steady
> linear-winding limit cycle, cc-independent. The *mechanistic why* is **open** → requires the bifurcation
> analysis (deferred Step B). Original (now-superseded) text kept below for the record.

### (superseded) Why B4 is the only escape: anti-dominance selection feedback

**k-law** (`simulation/personality_coupling.py:23-27`): `k_eff = k_base · [1 + β·(1/(3·dominance) − 1)]`,
dominance ∈ [⅓,1]. ⇒ selection strength is **reduced when a strategy dominates** (dominance→1 ⇒ factor 1−⅔β),
**full when balanced** (dominance=⅓ ⇒ factor 1). An **anti-dominance / anti-runaway feedback** that weakens
selection exactly when the trajectory heads for a simplex corner.

**Causal (β)-sweep on observable `p_`, k_base=0.08** (window-scaling ratio net(2000)/net(250); linear≈8):

| β | wind+ | net/1k | p_amp | scaling ratio | reading |
|---|---|---|---|---|---|
| 0.00 (fixed-k control) | 0/3 | −6.7 | 0.140 | −1.8 / −1.7 / +2.7, quarters sign-flip | **random walk** (no rotation) |
| **0.30** | **3/3** | −4.0 | **0.239** | **+7.8 / +8.0 / +7.3**, quarters all ~−0.9 | **steady rotation** |
| 0.60 | 1/3 | −2.0 | 0.183 | — | partial (near floor) |
| 1.00 | 0/3 | −2.4 | 0.170 | — | **over-damped** (net→0) |

**Same generator, only β changes ⇒ random-walk → steady constant-rate rotation.** β=0.3 also yields the
*largest* amplitude (the feedback sustains a bigger orbit); β≥0.6 over-damps. The orbit lives in the
observable proportions (linear winding ~−3.7 turns/1000) yet turning-consistency reads 0.51 because
finite-population jitter dominates the step-local cross-products.

**Conclusion:** B4's escape is **anti-dominance selection feedback as a phase-preserving mechanism** that
sustains observable RPS rotation under sampling — invisible to step-local rotation metrics, visible to
integrated winding. This converts the negative result into *"a negative result with a localized,
mechanistically-explained escape."*

**Residuals:** (1) short-scout β=0.6_k0.08 was 1/3 wind+ but the Phase-2 10-seed β=0.6_k0.08 was 10/10
(−3.07±0.027) — the within-condition wind+ rate is sensitive near the floor; cleanest exemplars =
β=0.3_k0.08 (3/3) + Phase-2 β=0.6 (10/10). (2) Need k_base=0.08 (k0.06 conditions don't pass) — a selection-
strength threshold not yet explained. (3) The *dynamical* reason moderate-β sustains while high-β damps
(a bifurcation/Hopf analysis) is open.

## Synthesis — a 4-way taxonomy, not a uniform negative result

| cell | step-local turning-consistency | integrated winding | reading |
|---|---|---|---|
| Arm A mean-field | high (1.0) | high+coherent | genuine rotation |
| b5 keystone / c1 | chance (0.51) | none | **genuinely no rotation** (sampling killed it / over-homogenized) |
| t minibatch | `w_` 0.96 (artifact) | `w_` coherent but **negligible amplitude**; `p_` not robust | microscopic / not observable |
| **b4 state-k** | `p_` 0.51 (**misses it**) | `p_` **robust steady rotation** | **metric-disagreement cell** |

## Implication for the claim (decision PENDING — user's call)

- "**L3 unreachable**" is **not** defensible as a general statement. The supportable form is
  **"L3 unreachable under the step-local turning-consistency criterion."**
- The B5 keystone causal story holds (sampling destroys rotation *for that generator*, on both metrics).
- But **b4 (a sampled cell) reaches observable L3 rotation under a winding criterion** — so the bottleneck is
  not "sampling destroys rotation," it is "sampling jitter defeats *step-local* rotation metrics even when
  integrated winding survives into the observable proportions."

### Open decisions (do NOT proceed to draft_v2 before these)
1. **Formal definition of "rotation"** for the paper: (i) turning-consistency only, (ii) winding-number only,
   or (iii) report both and make the *metric-disagreement* the contribution. b4 forces the choice.
2. Binding series (`p_` observable vs `w_` latent) — secondary to (1); recommend `p_` headline / `w_` mechanism
   **iff** (1) resolves toward turning-consistency-with-winding-caveat.
