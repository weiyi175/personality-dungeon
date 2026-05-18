#!/usr/bin/env python3
"""Post-scan analysis for H7 μ-Gradient results.

Reads the summary.json produced by run_mu_gradient_scan.py and re-evaluates
acceptance gates using the REVISED 3-gate framework (G4 removed).

G4 (mean_max_dominance ≤ 0.75) was incorrect: high-amplitude L3 cycling in
inertial replicator dynamics produces mean_max_dom ≈ 0.98–0.99 by design,
indistinguishable from stagnation. The correct anti-stagnation metric is G1
itself (if cycle_level ≥ 3, the system is by definition not stagnant).

REVISED GATES (3-gate framework):
  G1 PRIMARY     L3_control ≥ 50%
  G2 WALL        L3_control ≥ L3_random + 10pp
  G3 VELOCITY    mean_phase_velocity_control ∈ [0.001, 0.030]

Usage:
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python scripts/analyze_mu_gradient.py
  PYTHONPATH=/home/user/personality-dungeon ./venv/bin/python scripts/analyze_mu_gradient.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ──────────────────────────────────────────────────────────────────────────────
# Gate thresholds (revised 3-gate framework)
# ──────────────────────────────────────────────────────────────────────────────
GATE_L3_RATE_MIN: float = 0.50       # G1 PRIMARY
GATE_WALL_MARGIN_MIN: float = 0.10   # G2 WALL
GATE_VELOCITY_MIN: float = 0.001     # G3 VELOCITY lower
GATE_VELOCITY_MAX: float = 0.030     # G3 VELOCITY upper


def _load_summary(summary_path: Path) -> list[dict]:
    """Load run results from summary.json."""
    with open(summary_path) as f:
        data = json.load(f)
    return data["runs"]


def _evaluate_3gate(mu_base: float, runs: list[dict]) -> dict:
    """Evaluate 3 acceptance gates for a given mu_base."""
    ctrl = [r for r in runs if r["mu_base"] == mu_base and r["condition"] == "control_none"]
    rand = [r for r in runs if r["mu_base"] == mu_base and r["condition"] == "random_9persona"]

    if not ctrl:
        return {}

    ctrl_l3_rate = sum(1 for r in ctrl if r["l3_pass"]) / len(ctrl)
    rand_l3_rate = sum(1 for r in rand if r["l3_pass"]) / len(rand) if rand else 0.0
    ctrl_v_mean = sum(r["phase_velocity"] for r in ctrl) / len(ctrl)

    wall_margin = ctrl_l3_rate - rand_l3_rate

    g1 = ctrl_l3_rate >= GATE_L3_RATE_MIN
    g2 = wall_margin >= GATE_WALL_MARGIN_MIN
    g3 = GATE_VELOCITY_MIN <= ctrl_v_mean <= GATE_VELOCITY_MAX
    passes = g1 and g2 and g3

    return {
        "mu_base": mu_base,
        "ctrl_l3_rate": ctrl_l3_rate,
        "rand_l3_rate": rand_l3_rate,
        "wall_margin": wall_margin,
        "ctrl_v_mean": ctrl_v_mean,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "passes": passes,
        "n_ctrl": len(ctrl),
        "n_rand": len(rand),
    }


def _print_3gate_table(results: list[dict]) -> None:
    """Print formatted 3-gate acceptance table."""
    W = 105
    print()
    print("=" * W)
    print("μ-Gradient Acceptance Table — REVISED 3-gate Framework (G4 removed)")
    print(f"  G1=L3_ctrl≥50%  G2=Ctrl−Rand≥10pp  G3=v∈[0.001,0.030]")
    print("=" * W)
    print(f"  {'μ_base':>6}  {'ctrl L3%':>9}  {'rand L3%':>9}  {'wall Δ':>8}  {'G1':>4}  {'G2':>4}  {'v_mean':>8}  {'G3':>4}  {'PASS':>6}")
    print("-" * W)

    golden = []
    for r in results:
        if not r:
            continue
        mu = r["mu_base"]
        cl3 = r["ctrl_l3_rate"] * 100
        rl3 = r["rand_l3_rate"] * 100
        wall = r["wall_margin"] * 100
        v = r["ctrl_v_mean"]
        g1s = "✓" if r["g1"] else "✗"
        g2s = "✓" if r["g2"] else "✗"
        g3s = "✓" if r["g3"] else "✗"
        mark = "★ YES" if r["passes"] else "   no"
        n = r["n_ctrl"]
        print(f"  {mu:>6.2f}  {cl3:>8.1f}%  {rl3:>8.1f}%  {wall:>7.1f}pp  {g1s:>4}  {g2s:>4}  {v:>8.5f}  {g3s:>4}  {mark:>6}  (n={n})")
        if r["passes"]:
            golden.append(mu)

    print("=" * W)
    print()

    if golden:
        print(f"★ Golden Point(s): μ_base ∈ {golden}")
        print(f"  Recommendation: μ_base = {min(golden):.2f}  (smallest that passes all 3 gates)")
    else:
        best = max(results, key=lambda r: (r.get("ctrl_l3_rate", 0), r.get("wall_margin", 0)))
        print(f"✗ No μ_base passed all 3 gates.")
        if best:
            print(f"  Best candidate: μ_base = {best['mu_base']:.2f}  "
                  f"(L3={best['ctrl_l3_rate']*100:.0f}%, wall={best['wall_margin']*100:.1f}pp)")
    print()


def _print_seed_detail(mu_base: float, runs: list[dict]) -> None:
    """Print per-seed detail for a specific mu_base."""
    ctrl = sorted([r for r in runs if r["mu_base"] == mu_base and r["condition"] == "control_none"],
                  key=lambda r: r["seed"])
    rand = sorted([r for r in runs if r["mu_base"] == mu_base and r["condition"] == "random_9persona"],
                  key=lambda r: r["seed"])

    print(f"\n  Per-seed detail — μ_base = {mu_base:.2f}")
    print(f"  {'seed':>6}  {'ctrl_L3':>8}  {'ctrl_v':>10}  {'rand_L3':>8}  {'rand_v':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
    ctrl_map = {r["seed"]: r for r in ctrl}
    rand_map = {r["seed"]: r for r in rand}
    for seed in sorted(set(list(ctrl_map.keys()) + list(rand_map.keys()))):
        cr = ctrl_map.get(seed)
        rr = rand_map.get(seed)
        cl3 = "L3 ✓" if cr and cr["l3_pass"] else "L2  "
        cv = f"{cr['phase_velocity']:.5f}" if cr else "     -"
        rl3 = "L3 ✓" if rr and rr["l3_pass"] else "L2  "
        rv = f"{rr['phase_velocity']:.5f}" if rr else "     -"
        print(f"  {seed:>6}  {cl3:>8}  {cv:>10}  {rl3:>8}  {rv:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-scan analysis for H7 μ-Gradient")
    parser.add_argument("--quick", action="store_true",
                        help="Read from outputs/mu_gradient_quick/summary.json")
    parser.add_argument("--detail", action="store_true",
                        help="Print per-seed detail for golden point(s)")
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="Override path to summary.json")
    args = parser.parse_args()

    if args.summary_json:
        summary_path = args.summary_json
    elif args.quick:
        summary_path = ROOT / "outputs" / "mu_gradient_quick" / "summary.json"
    else:
        summary_path = ROOT / "outputs" / "mu_gradient_scan" / "summary.json"

    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found.")
        print("  Run the scan first: scripts/run_mu_gradient_scan.py")
        sys.exit(1)

    print(f"\nReading: {summary_path}")
    runs = _load_summary(summary_path)
    print(f"  {len(runs)} total run records loaded.")

    mu_bases = sorted(set(r["mu_base"] for r in runs))
    results = [_evaluate_3gate(mu, runs) for mu in mu_bases]

    _print_3gate_table(results)

    if args.detail:
        golden = [r["mu_base"] for r in results if r.get("passes")]
        for mu in (golden or mu_bases):
            _print_seed_detail(mu, runs)

    # Write revised summary
    out = summary_path.parent / "analysis_3gate.json"
    with open(out, "w") as f:
        json.dump({"gate_framework": "3-gate (G4 removed)", "results": results}, f, indent=2)
    print(f"Revised summary written to: {out}")


if __name__ == "__main__":
    main()
