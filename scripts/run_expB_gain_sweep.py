"""Exp B — does the authored 3-faction RPS payoff coexist or rotate, and where's the flip?

Reuses the replicator machinery from ecology_replicator_probe.py but SWEEPS the
gain instead of running one locked point. Payoff (evolution.independent_rl):
    A = [[0, a-cr, -b], [-b-cr, 0, a], [a+cr, -b+cr, 0]]
  a  = cross-WIN  (beating your prey)   -- "combat 做爽" pushes this up
  b  = cross-LOSS (losing to predator)  -- "combat 做爽" (symmetric decisive counter) pushes this up too
  cr = directional cross-coupling

DV = max real part of tangent-space eigenvalues at the interior fixed point q*:
  maxRe < 0  -> stable focus  = damped spiral-IN = COEXISTENCE (diversity, ecology-healthy)
  maxRe > 0  -> unstable      = spiral-OUT / heteroclinic = ROTATION (monoculture whiplash, feared)
Flip boundary (maxRe=0) = the precise gain where combat-strong tips ecology into rotation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT))
from evolution.independent_rl import strategy_payoff_matrix, STRATEGY_SPACE

OUT_DIR = ROOT / "outputs" / "expB_gain_sweep"
A_FIXED = 1.0
B_VALS = [round(x, 3) for x in np.linspace(0.6, 1.4, 17)]
CR_VALS = [0.0, 0.10, 0.20, 0.35, 0.50]
TOL = 1e-6


def interior_fixed_point(A):
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1)); M[:n, :n] = A; M[:n, n] = -1.0; M[n, :n] = 1.0
    rhs = np.zeros(n + 1); rhs[n] = 1.0
    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None
    q = sol[:n]
    if np.any(q < -1e-9) or np.any(q > 1 + 1e-9):
        return None
    return np.clip(q, 0, 1)


def replicator_rhs(q, A):
    f = A @ q
    return q * (f - q @ f)


def jacobian_at(q, A, eps=1e-6):
    n = A.shape[0]; J = np.zeros((n, n))
    for j in range(n):
        dq = np.zeros(n); dq[j] = eps
        J[:, j] = (replicator_rhs(q + dq, A) - replicator_rhs(q - dq, A)) / (2 * eps)
    return J


def tangent_eigs(J):
    n = J.shape[0]
    ones = np.ones(n) / np.sqrt(n)
    B, _ = np.linalg.qr(np.eye(n) - np.outer(ones, ones))
    B = B[:, :n - 1]
    return np.linalg.eigvals(B.T @ J @ B)


def cell(b, cr):
    A = np.array(strategy_payoff_matrix(a=A_FIXED, b=b, cross=cr))
    q = interior_fixed_point(A)
    if q is None:
        return {"b": b, "cr": cr, "q_star": None, "maxRe": None, "has_imag": None, "min_share": None}
    eigs = tangent_eigs(jacobian_at(q, A))
    return {
        "b": b, "cr": cr,
        "q_star": [round(float(x), 3) for x in q],
        "maxRe": float(eigs.real.max()),
        "has_imag": bool(np.any(np.abs(eigs.imag) > 1e-6)),
        "min_share": float(q.min()),
    }


def flip_b(cells_for_cr):
    """linear-interpolate the b where maxRe crosses 0 (coexist->rotate), ascending b."""
    pts = [(c["b"], c["maxRe"]) for c in cells_for_cr if c["maxRe"] is not None]
    for (b0, r0), (b1, r1) in zip(pts, pts[1:]):
        if r0 < 0 <= r1 or r0 <= 0 < r1:
            return round(b0 + (0 - r0) * (b1 - b0) / (r1 - r0), 4)
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = {cr: [cell(b, cr) for b in B_VALS] for cr in CR_VALS}

    print("strategy space:", list(STRATEGY_SPACE), " a fixed =", A_FIXED)
    print("\nmaxRe sign grid  (− = COEXIST/spiral-in,  + = ROTATE/spiral-out)  rows=b, cols=cr")
    print("   b  | " + "  ".join(f"cr={cr:.2f}" for cr in CR_VALS))
    for i, b in enumerate(B_VALS):
        cells_row = []
        for cr in CR_VALS:
            c = grid[cr][i]
            if c["maxRe"] is None:
                cells_row.append(" noFP ")
            else:
                sign = "-" if c["maxRe"] < -TOL else ("+" if c["maxRe"] > TOL else "0")
                cells_row.append(f"{sign}{abs(c['maxRe']):.3f}")
        print(f" {b:.2f} | " + "  ".join(cells_row))

    print("\nflip b* (maxRe crosses 0; coexist for b<b*, rotate for b>b*):")
    flips = {}
    for cr in CR_VALS:
        fb = flip_b(grid[cr])
        flips[cr] = fb
        print(f"  cr={cr:.2f}:  b* = {fb}   (a=1.0, so flip at b/a = {fb})")

    # locked ecology point cross-check (probe uses a=1,b=0.9,cr=0.2)
    locked = cell(0.9, 0.20)
    print(f"\nlocked ecology point (a=1,b=0.9,cr=0.2): maxRe={locked['maxRe']:+.4f} "
          f"q*={locked['q_star']} -> {'COEXIST' if locked['maxRe'] < 0 else 'ROTATE'}")

    json.dump({"a_fixed": A_FIXED, "b_vals": B_VALS, "cr_vals": CR_VALS,
               "grid": grid, "flip_b_per_cr": flips, "locked_point": locked},
              open(OUT_DIR / "expB_summary.json", "w"), indent=2)
    print(f"\nwritten: {(OUT_DIR / 'expB_summary.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
