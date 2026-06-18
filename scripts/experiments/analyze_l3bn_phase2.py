#!/usr/bin/env python3
"""L3-BN Phase-2 confirmatory analysis (pre-registered).

Locked plan (docs/experiments/l3_bottleneck/L3_BOTTLENECK_PREREGISTRATION.md):
- H0-neg:  per-cell L3-seed rate (cycle_level==3). Tolerance <=1/10 = tail blip; >=2/10 refutes.
- H1-chance: per-cell one-sample t-test of stage3_score(=consistency) vs mu0=0.50 (two-sided),
             with 95% CI and Cohen's d vs 0.50.
- H2-threshold-robust: recompute L3-seed rate at eta in {0.55,0.60,0.65,0.70} from consistency.
- H-PC: mean-field positive control g1_mean_field must hold cycle_level==3, consistency==1.0.

Reads the Phase-2 per-cell summary TSVs and emits a JSON + console report.
Read-only over data; no re-simulation.
"""
from __future__ import annotations
import csv, json, math, statistics as st
from pathlib import Path

P2 = Path("reports/experiments/l3_bottleneck/phase2")
MU0 = 0.50
ETAS = [0.55, 0.60, 0.65, 0.70]

# (label, summary_tsv, condition, role)
CELLS = [
    ("B5_keystone_sampled",  "b5_summary.tsv", "g2_sampled_delta0p000",          "Arm B keystone (sampled twin of mean-field control)"),
    ("B4_state_k",           "b4_summary.tsv", "beta0p60_k0p08",                 "Arm B (closest Phase-1 approach to bar)"),
    ("T_local_minibatch",    "t_summary.tsv",  "t_lattice4_minibatch",           "Arm B (only valid local-replicator test)"),
    ("C1_pairwise_std",      "c1_summary.tsv", "g2_c1_small_world_b10p0_m0p5",   "Arm B (locked 3000/1000/1000 window)"),
    ("C1_pairwise_native",   "c1_native_summary.tsv", "g2_c1_small_world_b10p0_m0p5", "C1 supplementary, native 5000/1500/1500 window"),
]
CONTROL_MF = ("B5_meanfield_control", "b5_summary.tsv", "g1_mean_field_delta0p000", "Arm A positive control (deterministic mean-field)")


def _load(tsv: str, condition: str) -> list[dict]:
    p = P2 / tsv
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p), delimiter="\t"))
    return [r for r in rows if r.get("condition") == condition and r.get("stage3_score") not in (None, "")]


def _t_vs_mu0(xs: list[float], mu0: float) -> dict:
    n = len(xs)
    m = st.mean(xs)
    sd = st.pstdev(xs) if n > 1 else 0.0
    s = st.stdev(xs) if n > 1 else 0.0
    se = (s / math.sqrt(n)) if (n > 1 and s > 0) else 0.0
    t = ((m - mu0) / se) if se > 0 else float("nan")
    # 95% CI via normal approx (n=10 small; report t-crit ~2.262 for df=9)
    tcrit = 2.262
    ci = (m - tcrit * se, m + tcrit * se) if se > 0 else (m, m)
    d = ((m - mu0) / s) if s > 0 else float("nan")
    return {"n": n, "mean": m, "sd_sample": s, "se": se, "t_vs_0.50": t,
            "ci95": list(ci), "cohens_d_vs_0.50": d, "ci_covers_0.50": ci[0] <= mu0 <= ci[1] if se > 0 else None}


def analyze_cell(label, tsv, cond, role) -> dict:
    rows = _load(tsv, cond)
    if not rows:
        return {"label": label, "condition": cond, "status": "MISSING", "source": tsv}
    cons = [float(r["stage3_score"]) for r in rows]
    levels = [int(r["cycle_level"]) for r in rows]
    l3 = sum(1 for L in levels if L == 3)
    eta_sweep = {f"eta_{e}": sum(1 for c in cons if c >= e) for e in ETAS}
    stat = _t_vs_mu0(cons, MU0)
    verdict = "negative_holds" if l3 <= 1 else "REFUTED"
    return {
        "label": label, "condition": cond, "role": role, "source": tsv,
        "n": len(rows), "L3_seed_count": l3, "tolerance_rule": "<=1 blip / >=2 refutes",
        "verdict": verdict,
        "levels_observed": sorted(set(levels)),
        "consistency": {"min": min(cons), "max": max(cons), **stat},
        "L3_rate_by_eta": eta_sweep,
    }


def main() -> None:
    out = {"mu0": MU0, "etas": ETAS, "cells": [], "positive_control": None}
    for c in CELLS:
        out["cells"].append(analyze_cell(*c))
    # positive control
    rows = _load(CONTROL_MF[1], CONTROL_MF[2])
    if rows:
        cons = [float(r["stage3_score"]) for r in rows]
        lv = [int(r["cycle_level"]) for r in rows]
        out["positive_control"] = {
            "label": CONTROL_MF[0], "condition": CONTROL_MF[2], "n": len(rows),
            "all_L3": all(L == 3 for L in lv), "all_consistency_1.0": all(abs(c - 1.0) < 1e-9 for c in cons),
            "consistency_min": min(cons), "consistency_max": max(cons), "levels": sorted(set(lv)),
            "verdict": "positive_control_holds" if (all(L == 3 for L in lv) and all(abs(c - 1.0) < 1e-9 for c in cons)) else "CONTROL_FAILED",
        }
    P2.mkdir(parents=True, exist_ok=True)
    (P2 / "phase2_confirmatory_analysis.json").write_text(json.dumps(out, indent=2))

    # console
    print("=" * 78)
    print("L3-BN Phase-2 confirmatory analysis")
    print("=" * 78)
    pc = out["positive_control"]
    if pc:
        print(f"[Arm A positive control] {pc['condition']}: n={pc['n']} "
              f"all_L3={pc['all_L3']} all_consistency=1.0:{pc['all_consistency_1.0']} -> {pc['verdict']}")
    print("-" * 78)
    for c in out["cells"]:
        if c.get("status") == "MISSING":
            print(f"[{c['label']}] MISSING ({c['source']})"); continue
        cc = c["consistency"]
        print(f"[{c['label']}] {c['condition']}")
        print(f"    L3={c['L3_seed_count']}/{c['n']}  levels={c['levels_observed']}  verdict={c['verdict']}")
        print(f"    consistency mean={cc['mean']:.4f} sd={cc['sd_sample']:.4f} 95%CI={cc['ci95'][0]:.4f}..{cc['ci95'][1]:.4f} "
              f"covers0.50={cc['ci_covers_0.50']} d_vs0.50={cc['cohens_d_vs_0.50']:.2f}")
        print(f"    L3-rate by eta: " + "  ".join(f"{k}={v}" for k, v in c["L3_rate_by_eta"].items()))
    print("=" * 78)


if __name__ == "__main__":
    main()
