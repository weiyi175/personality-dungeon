"""Scan parameters to approach the Hopf boundary (near-neutral rotation).

This helper is meant for *next-step* research iteration:
- You already confirmed directional rotation (Level3) and complex eigen-structure.
- Now you want to move from "damped spiral" to "near-neutral" (or unstable) by
  tuning parameters so that:
    ODE mode: alpha = Re(\lambda) ~ 0
    Lagged mode: rho = max |\lambda| ~ 1

No simulation is involved; this is local linearization around the uniform
interior equilibrium p*=(1/3,1/3,1/3).

Run:
  python -m analysis.hopf_scan --mode ode --scan b --a 0.385 --b-min 0.05 --b-max 0.25
  python -m analysis.hopf_scan --mode lagged --scan b --a 0.385 --b-min 0.05 --b-max 0.25 --selection-strength 0.02
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from analysis.jacobian_rotation import (
    lagged_jacobian_at_uniform,
    ode_jacobian_at_uniform,
    summarize_lagged,
    summarize_ode,
)


@dataclass(frozen=True)
class ScanPoint:
    param: float
    metric: float  # alpha (ODE) or (rho-1) (lagged)
    omega: float | None
    label: str


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [float(a)]
    aa = float(a)
    bb = float(b)
    step = (bb - aa) / float(n - 1)
    return [aa + step * i for i in range(int(n))]


def _bisect_root(f, lo: float, hi: float, *, iters: int = 40) -> tuple[float, float, float]:
    """Return (mid, f(mid), width) assuming f(lo) and f(hi) bracket 0."""
    a = float(lo)
    b = float(hi)
    fa = float(f(a))
    fb = float(f(b))
    if fa == 0.0:
        return (a, fa, 0.0)
    if fb == 0.0:
        return (b, fb, 0.0)
    if fa * fb > 0:
        raise ValueError("Root not bracketed")

    for _ in range(int(iters)):
        m = 0.5 * (a + b)
        fm = float(f(m))
        if fm == 0.0:
            return (m, fm, abs(b - a))
        if fa * fm <= 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    m = 0.5 * (a + b)
    fm = float(f(m))
    return (m, fm, abs(b - a))


def main() -> None:
    p = argparse.ArgumentParser(description="Scan (a,b) to approach Hopf boundary near uniform equilibrium")
    p.add_argument("--mode", choices=["ode", "lagged"], default="ode")
    p.add_argument("--scan", choices=["a", "b"], default="b", help="Which parameter to scan")
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--b", type=float, required=True)

    p.add_argument("--a-min", type=float, default=None)
    p.add_argument("--a-max", type=float, default=None)
    p.add_argument("--b-min", type=float, default=None)
    p.add_argument("--b-max", type=float, default=None)
    p.add_argument("--n", type=int, default=41)

    p.add_argument("--selection-strength", type=float, default=0.02)
    p.add_argument("--fd-h", type=float, default=1e-6)
    p.add_argument("--summary-tol", type=float, default=1e-6)

    p.add_argument("--refine", action="store_true", help="If a sign change is found, bisect to refine the boundary")
    p.add_argument("--refine-iters", type=int, default=50)

    args = p.parse_args()

    mode = str(args.mode)
    scan = str(args.scan)
    a0 = float(args.a)
    b0 = float(args.b)
    h = float(args.fd_h)
    tol = float(args.summary_tol)
    k = float(args.selection_strength)

    if scan == "a":
        lo = float(args.a_min if args.a_min is not None else a0)
        hi = float(args.a_max if args.a_max is not None else a0)
    else:
        lo = float(args.b_min if args.b_min is not None else b0)
        hi = float(args.b_max if args.b_max is not None else b0)

    if lo == hi:
        raise SystemExit("Need a non-degenerate scan range: set --a-min/--a-max or --b-min/--b-max")

    grid = _linspace(lo, hi, int(args.n))

    def metric_at(param: float) -> ScanPoint:
        if scan == "a":
            aa, bb = float(param), float(b0)
        else:
            aa, bb = float(a0), float(param)

        if mode == "ode":
            rep = ode_jacobian_at_uniform(a=aa, b=bb, h=h)
            s = summarize_ode(rep, tol=tol)
            # root target: alpha ~ 0
            return ScanPoint(param=float(param), metric=float(s.alpha), omega=float(s.omega), label=s.stability)

        rep2 = lagged_jacobian_at_uniform(a=aa, b=bb, selection_strength=k, h=h)
        s2 = summarize_lagged(rep2, tol=tol)
        # root target: rho-1 ~ 0
        return ScanPoint(param=float(param), metric=float(s2.rho - 1.0), omega=None, label=s2.stability)

    pts = [metric_at(x) for x in grid]

    title_metric = "alpha" if mode == "ode" else "rho-1"
    print("=== Hopf scan (local linearization at p*) ===")
    print(f"mode={mode} scan={scan} range=[{lo:g},{hi:g}] n={int(args.n)}")
    print(f"base a={a0:g} b={b0:g} selection_strength={k:g} fd_h={h:g} tol={tol:g}")

    if mode == "ode":
        print("\nparam\talpha\t\tomega\t\tlabel")
        for pt in pts:
            assert pt.omega is not None
            print(f"{pt.param:.8g}\t{pt.metric:+.6g}\t{pt.omega:.6g}\t{pt.label}")
    else:
        print("\nparam\trho-1\t\tlabel")
        for pt in pts:
            print(f"{pt.param:.8g}\t{pt.metric:+.6g}\t{pt.label}")

    best = min(pts, key=lambda q: abs(q.metric))
    print(f"\nclosest |{title_metric}| at {scan}={best.param:.8g}: {title_metric}={best.metric:+.6g} ({best.label})")

    # Try to find a bracketing interval.
    bracket: tuple[float, float] | None = None
    for p1, p2 in zip(pts[:-1], pts[1:]):
        if p1.metric == 0.0:
            bracket = (p1.param, p1.param)
            break
        if p1.metric * p2.metric < 0.0:
            bracket = (p1.param, p2.param)
            break

    if bracket is None:
        print("no sign change found on this scan; widen range or scan a different axis")
        return

    lo2, hi2 = bracket
    if lo2 == hi2:
        print(f"exact root hit at {scan}={lo2:.8g}")
        return

    print(f"bracket found: {scan} in [{lo2:.8g}, {hi2:.8g}]")
    if not bool(args.refine):
        print("(use --refine to bisect and refine this boundary)")
        return

    def f(z: float) -> float:
        return float(metric_at(z).metric)

    mid, fmid, width = _bisect_root(f, lo2, hi2, iters=int(args.refine_iters))
    # Re-evaluate for label/omega at midpoint.
    mid_pt = metric_at(mid)
    if mode == "ode":
        assert mid_pt.omega is not None
        print(
            f"refined ~root: {scan}={mid:.10g}  alpha={mid_pt.metric:+.6g}  omega={mid_pt.omega:.6g}  width~{width:.3g}  ({mid_pt.label})"
        )
    else:
        print(
            f"refined ~root: {scan}={mid:.10g}  rho-1={mid_pt.metric:+.6g}  width~{width:.3g}  ({mid_pt.label})"
        )


if __name__ == "__main__":
    main()
