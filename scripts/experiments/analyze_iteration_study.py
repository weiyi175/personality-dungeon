#!/usr/bin/env python
"""
Personality-iteration study analysis (iterated vs reset across the 3-cycle run).

Consumes the player-test tracker + survey JSON (the same files the live apparatus
writes via tracker.save() / survey.save()) and runs the pre-registered analysis
for the iteration study. See `人格迭代實驗_規劃_v1.md`.

Selection
---------
Only sessions carrying the iteration-study fields are analysed:
  run_id != "" AND iteration_arm in {"iterated","reset"}.
Legacy P7-H sessions (no run_id) are ignored. Sessions are grouped by run_id into
per-participant runs and ordered by cycle_index (0/1/2).

Dependent variables
-------------------
  M  (manipulation check) : per run, u_n = unit(will_personality_vector − B);
                            anchor = cos(u_first, u_last). Predict iterated > reset
                            (iterated keeps early wills' direction → more anchored).
  E2 (dynamics, exploratory): per (arm, cycle_index) summaries of max_proximity /
                            total_displacement / rounds-survived (n_steps).
  q4 (EXPLORATORY floor-check, NOT confirmatory): q4_continuity (survey, joined per run
                            by the run's session_ids). Manipulation-sufficiency probe:
                            expect BOTH arms to floor (narrative-only priming can't move
                            felt continuity since the mechanism is invisible). A both-floor
                            null is INFORMATIVE — it mandates a decoupled continuity v2,
                            not a failed hypothesis. Demoted from secondary 2026-06-13.

Read-only: never writes to the live session files; emits a summary JSON to --out.

Usage
-----
  ./venv/bin/python scripts/experiments/analyze_iteration_study.py \
      [--sessions reports/experiments/p7h_real_study/p7h_player_test_sessions.json] \
      [--survey   reports/experiments/p7h_real_study/p7h_survey_responses.json] \
      [--out      reports/experiments/p7h_real_study] [--min-steps 10]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _stats
except Exception:  # scipy optional; tests degrade to descriptive-only
    _stats = None

# Space-A baseline attractor (FEATURE_NAMES order) — must match
# DungeonLifecycleController.BASELINE_ATTRACTOR / bifurcation_detector.py.
BASELINE = np.array(
    [0.772, 0.724, 0.611, -0.923, -0.432, 0.289, 0.697, -0.471, 0.811]
)
ARMS = ("iterated", "reset")

# naive 受試者代碼樣式（P01, P02, …）。實驗者試玩用 "dev"、pre-pilot 既有資料用 "EXP_PREPILOT"
# → 都不符此樣式 → 被排除。participant_id allowlist 是 naive vs 實驗者的「權威」判別子，
# 因為 player_alias/is_human 在前端都 hardcoded、分不開（見 §8g）。
PILOT_PID_PATTERN = re.compile(r"^P\d{2,}$")


# ── loading ────────────────────────────────────────────────────────────────
def _items(blob, *keys):
    """Return the list of records from {key: {id: rec}} or a bare list/dict."""
    node = blob
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
            break
    if isinstance(node, dict):
        return list(node.values())
    return list(node) if isinstance(node, list) else []


def load_sessions(path: Path) -> list[dict]:
    return _items(json.loads(path.read_text()), "sessions")


def load_survey(path: Path) -> dict[str, dict]:
    """session_id -> survey response."""
    out = {}
    if not path.exists():
        return out
    for r in _items(json.loads(path.read_text()), "responses"):
        sid = r.get("session_id")
        if sid:
            out[sid] = r
    return out


# ── geometry ──────────────────────────────────────────────────────────────
def will_direction(sess: dict, field: str = "will_personality_vector") -> np.ndarray | None:
    """Unit vector of (field − B); None if unusable.

    field="will_personality_vector": the scaled RL-seed personality. For the
        iterated arm this is the *cumulative* B+Σdₖ, so cos(first,last) is anchored
        BY DESIGN — use only as a construction/pipeline check, not a behavioral one.
    field="will_sbert_vector": the raw single will the player wrote that cycle
        (freely rewritten; box is emptied each cycle — no seed/prefill). Identical
        computation for both arms → a genuine behavioral measure.
    """
    v = sess.get(field) or []
    if len(v) != 9:
        return None
    off = np.array(v, dtype=float) - BASELINE
    n = float(np.linalg.norm(off))
    if n < 1e-9:
        return None
    return off / n


# ── dynamics / saturation helpers ────────────────────────────────────────
def _climb_rate(traj: list[dict]) -> float | None:
    """Per-step proximity climb rate = OLS slope of proximity_after over step_index.
    Arm discriminator for E2 (max_proximity hits a ceiling ~1.0 in both arms)."""
    pts = [(int(t.get("step_index", i)), float(t["proximity_after"]))
           for i, t in enumerate(traj) if "proximity_after" in t]
    if len(pts) < 3:
        return None
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    if _stats is not None:
        return float(_stats.linregress(xs, ys).slope)
    dx = xs[-1] - xs[0]
    return float((ys[-1] - ys[0]) / dx) if dx else None


def _sat_index(reck) -> float | None:
    """Saturation = how far recklessness is pinned to a sigmoid extreme.
    1 - 2*min(r, 1-r): 0 at mid (0.5), 1 at either boundary (0 or 1). Higher =
    stronger cumulative pinning = larger 'mechanism dose'. (-1/None = unrecorded.)"""
    if reck is None:
        return None
    r = float(reck)
    if r < 0:
        return None
    return round(1.0 - 2.0 * min(r, 1.0 - r), 4)


def _spearman(xs: list, ys: list) -> dict:
    """Spearman rho on paired (x,y), dropping None/NaN. Robust to tiny N."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None
             and not (isinstance(x, float) and np.isnan(x))
             and not (isinstance(y, float) and np.isnan(y))]
    out = {"n_pairs": len(pairs)}
    if _stats is not None and len(pairs) >= 3:
        rho, p = _stats.spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        out["spearman_rho"] = round(float(rho), 4)
        out["p_two_sided"] = round(float(p), 4)
    else:
        out["note"] = "insufficient N (need >=3 paired iterated runs)"
    return out


# ── stats helpers ────────────────────────────────────────────────────────
def _dist(vals: list[float]) -> dict:
    """Per-arm distribution for N back-calculation. sd_upper95 = upper 95% CI on
    SD (chi-square): use THIS (not the point SD) to size the study — n≈10 pilot
    variance is noisy and would under-power (plan §6b lock 3)."""
    out = {"n": len(vals), "mean": round(float(np.mean(vals)), 4) if vals else None,
           "sd": None, "sd_upper95": None}
    if len(vals) >= 2:
        arr = np.array(vals, dtype=float)
        out["sd"] = round(float(arr.std(ddof=1)), 4)
        if _stats is not None:
            n = len(vals)
            var_upper = (n - 1) * float(arr.var(ddof=1)) / _stats.chi2.ppf(0.025, n - 1)
            out["sd_upper95"] = round(float(var_upper ** 0.5), 4)
    return out


def _mw(a: list[float], b: list[float]) -> dict:
    """Mann-Whitney U (a vs b) + rank-biserial effect size + per-arm dist
    (mean/sd/sd_upper95 for N sizing). Robust to tiny N."""
    out = {"n_a": len(a), "n_b": len(b),
           "median_a": float(np.median(a)) if a else None,
           "median_b": float(np.median(b)) if b else None,
           "dist_a": _dist(a), "dist_b": _dist(b)}
    if _stats is not None and len(a) >= 2 and len(b) >= 2:
        u, p = _stats.mannwhitneyu(a, b, alternative="two-sided")
        out["U"] = float(u)
        out["p_two_sided"] = float(p)
        out["rank_biserial"] = float(1.0 - 2.0 * u / (len(a) * len(b)))
    else:
        out["note"] = "insufficient N for test (need >=2 per arm)"
    return out


# ── analysis ─────────────────────────────────────────────────────────────
def analyse(sessions: list[dict], survey: dict[str, dict], min_steps: int,
            pilot_start: float = 0.0, pilot_participants: set[str] | None = None) -> dict:
    # group iteration-study sessions by run_id
    runs: dict[str, list[dict]] = {}
    n_legacy = 0
    for s in sessions:
        rid = s.get("run_id") or ""
        arm = s.get("iteration_arm") or ""
        if not rid or arm not in ARMS:
            n_legacy += 1
            continue
        runs.setdefault(rid, []).append(s)
    for rid in runs:
        runs[rid].sort(key=lambda x: (x.get("cycle_index", -1)))

    # ── participant_id gate (AUTHORITATIVE naive vs experimenter discriminator) ──
    # alias/is_human are hardcoded in the frontend → useless. A run is naive pilot data
    # ONLY if its participant_id ∈ allowlist (explicit set, else the P\d{2,} pattern).
    # "dev" (experimenter playtests) + "EXP_PREPILOT" (quarantined pre-pilot) never match.
    # Secondary safeguards (NOT authoritative): is_human all True AND started_at >= pilot_start.
    def _pid_in_allowlist(pid: str) -> bool:
        if pilot_participants is not None:
            return pid in pilot_participants
        return bool(PILOT_PID_PATTERN.match(pid))

    excluded = {"not_in_pilot_allowlist": [], "mixed_participant_id": [],
                "not_human": [], "before_pilot_start": []}
    kept: dict[str, list[dict]] = {}
    for rid, cs in runs.items():
        pids = sorted({str(c.get("participant_id") or "dev") for c in cs})
        if len(pids) > 1:
            excluded["mixed_participant_id"].append({"run_id": rid, "participant_ids": pids})
            continue
        pid = pids[0]
        if not _pid_in_allowlist(pid):
            excluded["not_in_pilot_allowlist"].append({"run_id": rid, "participant_id": pid})
            continue
        if not all(bool(c.get("is_human", False)) for c in cs):
            excluded["not_human"].append({"run_id": rid, "participant_id": pid})
            continue
        if pilot_start and min(float(c.get("started_at", 0.0)) for c in cs) < pilot_start:
            excluded["before_pilot_start"].append({"run_id": rid, "participant_id": pid})
            continue
        kept[rid] = cs
    n_excluded_runs = {k: len(v) for k, v in excluded.items()}
    n_naive_runs = len(kept)
    runs = kept  # downstream DVs see ONLY naive-allowlisted runs

    arm_of = {rid: (cs[0].get("iteration_arm")) for rid, cs in runs.items()}
    n_runs = {a: sum(1 for r in arm_of.values() if r == a) for a in ARMS}
    n_complete = {a: 0 for a in ARMS}
    n_partial = {a: 0 for a in ARMS}

    anchor_constr = {a: [] for a in ARMS}   # M:  cos over will_personality_vector (cumulative seed)
    anchor_behav = {a: [] for a in ARMS}    # E1: cos over will_sbert_vector (raw freely-written wills)
    q4 = {a: [] for a in ARMS}              # H_PI
    dyn = {a: {} for a in ARMS}             # E2: (arm, cycle) -> lists
    per_run = []                            # per-run records for saturation dose-response

    for rid, cycles in runs.items():
        arm = arm_of[rid]
        # Lock 2 (plan §6b): ONLY complete 3-cycle runs enter the DVs. A 2-cycle
        # dropout would make cos(first,last) = cos(cycle0,cycle1) — a different DV
        # that contaminates E1/M. Partials are counted but excluded from analysis.
        if len(cycles) < 3:
            n_partial[arm] += 1
            continue
        n_complete[arm] += 1

        m_val = None       # M construction cos
        e1_val = None      # E1 behavioral cos
        u0 = will_direction(cycles[0], "will_personality_vector")
        ul = will_direction(cycles[-1], "will_personality_vector")
        if u0 is not None and ul is not None:
            m_val = float(np.dot(u0, ul)); anchor_constr[arm].append(m_val)
        b0 = will_direction(cycles[0], "will_sbert_vector")
        bl = will_direction(cycles[-1], "will_sbert_vector")
        if b0 is not None and bl is not None:
            e1_val = float(np.dot(b0, bl)); anchor_behav[arm].append(e1_val)

        # E2: per-cycle dynamics. Primary discriminator = per-step climb rate
        # (max_proximity has a ceiling ~1.0 in both arms → poor discriminator).
        for c in cycles:
            ci = int(c.get("cycle_index", -1))
            if int(c.get("n_steps", 0)) < min_steps:
                continue
            d = dyn[arm].setdefault(ci, {"climb_rate": [], "max_proximity": [], "n_steps": []})
            cr = _climb_rate(c.get("trajectory", []))
            if cr is not None:
                d["climb_rate"].append(cr)
            d["max_proximity"].append(float(c.get("max_proximity", 0.0)))
            d["n_steps"].append(int(c.get("n_steps", 0)))

        # q4: run-level, attached to the FINAL (max cycle_index) session.
        final = cycles[-1]
        resp = survey.get(final.get("session_id"))
        rq4 = int(resp["q4_continuity"]) if (resp and int(resp.get("q4_continuity", 0)) > 0) else None
        if rq4 is not None:
            q4[arm].append(rq4)

        # saturation = recklessness of the FINAL (most-accumulated) cycle, pinned to a
        # sigmoid extreme. Mechanism "dose": how hard the cumulative drove personality.
        sat = _sat_index(final.get("will_recklessness"))
        per_run.append({"run_id": rid, "arm": arm, "E1": e1_val, "q4": rq4, "sat_index": sat})

    def _dyn_summary(arm):
        out = {}
        for ci, d in sorted(dyn[arm].items()):
            out[f"cycle_{ci}"] = {
                "n": len(d["n_steps"]),
                "climb_rate_per_step_mean": round(float(np.mean(d["climb_rate"])), 6) if d["climb_rate"] else None,
                "max_proximity_mean": round(float(np.mean(d["max_proximity"])), 4) if d["max_proximity"] else None,
                "rounds_survived_mean": round(float(np.mean(d["n_steps"])), 1) if d["n_steps"] else None,
            }
        return out

    # ── saturation covariate + within-iterated dose-response (exploratory) ──
    def _arm_sat_mean(a):
        vals = [r["sat_index"] for r in per_run if r["arm"] == a and r["sat_index"] is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    it = [r for r in per_run if r["arm"] == "iterated"]

    return {
        "selection": {
            "n_naive_pilot_runs": n_naive_runs,
            "n_iteration_runs": n_runs,
            "n_complete_3cycle_runs": n_complete,
            "n_partial_runs_excluded": n_partial,
            "n_legacy_sessions_ignored": n_legacy,
            "participant_gate": {
                "allowlist": (sorted(pilot_participants) if pilot_participants is not None
                              else f"pattern {PILOT_PID_PATTERN.pattern}"),
                "pilot_start": pilot_start or None,
                "n_runs_excluded_by_reason": n_excluded_runs,
                "excluded_runs": excluded,
            },
            "note": "participant_id allowlist is AUTHORITATIVE (alias/is_human hardcoded). "
                    "Only naive-allowlisted, complete 3-cycle runs enter M/E1/q4/E2/saturation "
                    "(lock 2). 'dev'/'EXP_PREPILOT' never match the allowlist.",
        },
        # Tier labels per LOCKED hierarchy 2026-06-13 (人格迭代實驗_規劃_v1.md §8d).
        "M_construction_check_QA": {
            "tier": "QA assertion (NOT a hypothesis)",
            "metric": "cos(first,last) of will_personality_vector (scaled RL seed). "
                      "Anchored BY DESIGN for iterated (cumulative) — tautological; only "
                      "verifies the accumulation pipeline ran. Excluded from power/claims.",
            **_mw(anchor_constr["iterated"], anchor_constr["reset"]),
        },
        "E1_will_writing_behavior_PRIMARY": {
            "tier": "PRIMARY confirmatory (two-sided; DIRECTION OPEN)",
            "metric": "cos(first,last) of will_sbert_vector (raw freely-written wills; "
                      "identical computation both arms). Joint mechanism+narrative effect "
                      "on writing coherence; no directional prediction (see §8d).",
            **_mw(anchor_behav["iterated"], anchor_behav["reset"]),
        },
        "q4_continuity_floor_check_exploratory": {
            "tier": "EXPLORATORY floor-check / manipulation-sufficiency (NOT confirmatory; demoted 2026-06-13)",
            "metric": "q4_continuity (1-10). Role: does pure-narrative priming move felt "
                      "continuity at all? Mechanism is visually invisible, so the ONLY cue is one "
                      "priming sentence.",
            "expected": "BOTH arms floor (low). This null is INFORMATIVE — it upgrades n=1 self-report "
                        "into n~20 citable evidence that narrative-only priming is insufficient, "
                        "mandating a decoupled continuity v2. Do NOT read a null as a failed hypothesis.",
            **_mw(q4["iterated"], q4["reset"]),
        },
        "E2_dynamics_exploratory": {
            "discriminator": "climb_rate_per_step_mean (max_proximity reported but ceiling-bound ~1.0)",
            "iterated": _dyn_summary("iterated"),
            "reset": _dyn_summary("reset"),
        },
        "saturation_covariate_exploratory": {
            "definition": "per run: 1-2*min(r,1-r), r=recklessness of FINAL (most-accumulated) "
                          "cycle. Higher = cumulative will pinned personality to a sigmoid "
                          "extreme = larger mechanism dose. Reset shown for contrast (its r is "
                          "the single final will, not cumulative). Mechanism frozen pre-pilot "
                          "(cap/renorm = v2 design question — see plan §8e).",
            "mean_by_arm": {a: _arm_sat_mean(a) for a in ARMS},
            "dose_response_within_iterated": {
                "rationale": "Within iterated, narrative priming is CONSTANT but saturation "
                             "(mechanism dose) VARIES across players. A positive slope = the "
                             "mechanism contributes above the constant narrative → first probe "
                             "at the claim ceiling; informs whether the v2 follow-up is worth it.",
                "caveat": "OBSERVATIONAL, non-random, confounded with trait-consistency "
                          "(players whose wills stay on-theme both saturate AND write coherently). "
                          "EXPLORATORY ONLY — does NOT replace the mechanism×narrative 2×2.",
                "sat_vs_E1_primary": _spearman([r["sat_index"] for r in it], [r["E1"] for r in it]),
                "sat_vs_q4_floor": _spearman([r["sat_index"] for r in it], [r["q4"] for r in it]),
            },
        },
    }


def main() -> None:
    base = Path("reports/experiments/p7h_real_study")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=Path, default=base / "p7h_player_test_sessions.json")
    ap.add_argument("--survey", type=Path, default=base / "p7h_survey_responses.json")
    ap.add_argument("--out", type=Path, default=base)
    ap.add_argument("--min-steps", type=int, default=10,
                    help="exclude abandoned cycles below this many steps from E2 dynamics")
    ap.add_argument("--pilot-start", type=float, default=0.0,
                    help="unix ts; secondary safeguard — runs starting before this are excluded")
    ap.add_argument("--pilot-participants", type=str, default="",
                    help="comma-separated explicit allowlist (e.g. P01,P02). "
                         "Empty → use default P\\d{2,} pattern.")
    args = ap.parse_args()

    pilot_participants = None
    if args.pilot_participants.strip():
        pilot_participants = {p.strip() for p in args.pilot_participants.split(",") if p.strip()}

    sessions = load_sessions(args.sessions)
    survey = load_survey(args.survey)
    result = analyse(sessions, survey, args.min_steps,
                     pilot_start=args.pilot_start, pilot_participants=pilot_participants)

    sel = result["selection"]
    n_iter = sel["n_iteration_runs"]   # naive-allowlisted only (post-gate)
    total = sum(n_iter.values())
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "iteration_study_analysis.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    gate = sel["participant_gate"]
    print(f"[GATE] participant allowlist={gate['allowlist']}  →  NAIVE pilot runs={sel['n_naive_pilot_runs']}  "
          f"| excluded runs by reason={gate['n_runs_excluded_by_reason']}")
    print(f"iteration-study NAIVE runs: iterated={n_iter['iterated']} reset={n_iter['reset']} "
          f"(complete 3-cycle: {sel['n_complete_3cycle_runs']}); "
          f"legacy ignored={sel['n_legacy_sessions_ignored']}")
    if total == 0:
        print("No NAIVE pilot runs yet — apparatus ready, experimenter/pre-pilot runs excluded, awaiting recruitment.")
    else:
        m = result["M_construction_check_QA"]
        e = result["E1_will_writing_behavior_PRIMARY"]
        h = result["q4_continuity_floor_check_exploratory"]
        print(f"[QA]      M construction median iter={m.get('median_a')} reset={m.get('median_b')} "
              f"p={m.get('p_two_sided', m.get('note'))}  (tautological, not a finding)")
        print(f"[PRIMARY] E1 will-write  median iter={e.get('median_a')} reset={e.get('median_b')} "
              f"p={e.get('p_two_sided', e.get('note'))}  (two-sided, direction open)")
        print(f"[FLOOR]   q4 continuity  median iter={h.get('median_a')} reset={h.get('median_b')} "
              f"p={h.get('p_two_sided', h.get('note'))}  (exploratory; both-floor null = informative)")
        e2 = result["E2_dynamics_exploratory"]
        def _cr(arm):
            return {k: v.get("climb_rate_per_step_mean") for k, v in e2[arm].items()}
        print(f"[E2]      climb-rate/step  iter={_cr('iterated')}  reset={_cr('reset')}")
        sat = result["saturation_covariate_exploratory"]
        dr = sat["dose_response_within_iterated"]
        print(f"[SAT]     mean_by_arm={sat['mean_by_arm']} | within-iterated "
              f"sat×E1 {dr['sat_vs_E1_primary'].get('spearman_rho', dr['sat_vs_E1_primary'].get('note'))} "
              f"| sat×q4 {dr['sat_vs_q4_floor'].get('spearman_rho', dr['sat_vs_q4_floor'].get('note'))}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
