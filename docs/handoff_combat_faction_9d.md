# Handoff prompt — 9D 人格更動 & 三簇（combat faction）分類

> Paste this into a fresh session to take over the **combat-faction / 9D-personality** branch.
> (This is the *game-build* lane — distinct from the *L3-bottleneck paper* lane.)
> Written 2026-06-21. Citations re-verified 2026-06-24 (all cited paths exist; `schemas.py:17-30`,
> RPS direction, endpoints, model dims, and the 13 pvp tests all confirmed). Still: treat claims as
> **to-be-verified against the cited files** before building on them, not gospel.

---

## Your role & the branch goal

You are continuing the design of **combat factions** for a personality-dungeon game. The core question
this branch has been answering: *given a fixed 9-trait personality vector, how should the 3 combat
factions be defined, do the traits themselves need changing, and how is the counter-matrix decided?*

## What was DECIDED (verify against sources before relying on)

1. **The 9 traits are fixed — do NOT change them.** `api/schemas.py:17-30` `PERSONALITY_BASIS`, git-set in
   `3dabc9a`, never modified. Grouped in code as 擴張組 Drivers (impulsiveness, assertiveness, optimism) /
   防禦組 Stabilizers (risk_aversion, suspicion, endurance) / 擾動組 Explorers (randomness, stability_seeking,
   curiosity). A trait-replacement (swap suspicion/optimism + retrain SBERT) was **considered and TERMINATED** —
   too costly, and the data didn't support it (see #3).
2. **Factions = behavioral projection, NOT a trait partition (Option 4+).** Real authored-will SBERT-9D
   structure is **2-axis**, not 3 clean clusters: PC1≈38% (bold↔cautious), PC2≈22% (optimism↔suspicion);
   **suspicion anchors PC2**. So define 3 factions as **3×120° argmax in the PC1–PC2 plane**, not by
   assigning 3 traits to each faction. (The code grouping and the 莽/野/穩 combat grouping are *imposed
   lenses*, not the data's shape.) Source: `scripts/run_expA_blast_radius.py` + memory `trait-blast-radius-expA`.
3. **The combat counter-matrix M must be AUTHORED, not measured.** H_counter was falsified (all 9 trait-buckets'
   fastest-collapse direction = +v1; proximity only measures the v1–v2 plane), so counters can't be read off
   the data — they're a design choice validated by playtest. One edge is data-backed (穩剋莽: recklessness→
   survival, Spearman ρ≈−0.98); 野剋穩 is authored fiction.
4. **3-archetype projection is a LOSSY, PC1-biased proxy of will diversity** (hides >50% of will diversity).
   So "archetype diversity" ≠ "personality diversity". Source: `reports/experiments/ecology_grain/GRAIN_FINDINGS.md`,
   `scripts/experiments/ecology_grain_analysis.py`.

## What is OPEN (the actual remaining work — build + playtest, not simulation)

- **Third-faction theme**: which trait direction the 3rd PC loads, and whether that set is *coherent,
  nameable, and fun* — only answerable by build+playtest, not analysis.
- **Authored M: arrows + values**, plus resolution mechanics (single/multi-round, in-session resistance,
  how much PvP reuses proximity, the effectiveness metric). All decided at build time.
- **Balance-test the authored M** on the already-built minimal combat loop: `api/pvp_manager.py` (added in
  `c373efc`: authored 3-RPS defensive▸aggressive▸balanced▸defensive; routes `/pvp/dungeons` +
  `/pvp/challenge` in `api/server.py:1753,1807`; `tests/test_pvp_manager.py` 13 green — re-confirmed 2026-06-24). The natural first experiment = sweep challenger scout-accuracy vs the
  dungeon's counter-owner policy → win-rate matrix + Rank-economy; check non-degeneracy (no dominant build,
  counters meaningful but not deterministic). This is the combat analog of the ecology Exp A/B.

## Pre-read (paths)

- `地牢設計_待回歸問題_parking_lot.md` **§A (Counter-policy/Combat)** — the canonical decided/open list.
- `地牢counter-policy_L0L1介面_規劃_v1.md` — counter-policy design.
- `api/schemas.py:17-30` — the 9 traits.
- `api/pvp_manager.py` + `tests/test_pvp_manager.py` — the built minimal combat loop.
- `reports/experiments/ecology_grain/GRAIN_FINDINGS.md` (+ `GRAIN_ANALYSIS.md`) — 3-archetype is lossy.
- `scripts/run_expA_blast_radius.py` — trait blast-radius / 2-axis PCA evidence.
- Auto-memory (read MEMORY.md index): `grain-archetype-hides-diversity`, `trait-blast-radius-expA`,
  `game-vision-original`, `apparatus-limits-static-personality`, `personality-ecology-layer`.

## Environment

- Primary dir `/home/user/personality-dungeon` (git repo). Python: `./venv/bin/python`.
- Backend `python -m api.server` has **no --reload** → restart after changing `api/`/`simulation/` or event JSON.
- Live SBERT model: frozen `BAAI/bge-base-zh-v1.5` (768-dim) + MLP head `outputs/mlp_opus_bgezh_v2.joblib`
  (`MultiOutputRegressor`, 768→9); SBERT is never fine-tuned. NEEDS a server restart to go live (see
  memory `sbert-9d-quality-step0`).
- Prefix shell commands with `rtk` (see `CLAUDE.md`).
- Frontend (Godot) lives at `/mnt/c/Users/n1166/personality-dungeon` (TABS, Windows).

## Recommended first action

Read parking-lot §A + `api/pvp_manager.py`, confirm the 13 tests still pass, then design/run the
**counter-matrix balance probe** (scout-accuracy × counter-owner policy → win-rate matrix + Rank economy).
Verify the "decided" claims above against their sources before building on them.
