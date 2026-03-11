r"""Jacobian + local rotation checks for `matrix_ab` (mean-field theory).

This module is intentionally *simulation-free*: it analyzes the deterministic
mean-field dynamics implied by the `matrix_ab` payoff and the exponential
replicator-style update.

Two related models are provided:

1) Replicator ODE (no lag)
   \dot x_i = x_i((A x)_i - x^T A x)

2) Lagged discrete update (matches current engine ordering under
   popularity_mode='expected' + exponential replicator update):
   - rewards at round t depend on previous popularity p_{t-1}
   - selection at round t is applied to the current sampling distribution p_t

    p_{t+1} \propto p_t \odot \exp(k * (u(p_{t-1}) - \bar u(p_t, p_{t-1})))

Where
  u(p_prev) = A p_prev
  \bar u(p_t,p_prev) = sum_i p_t[i] u_i(p_prev)

We work in 2D coordinates (u=x1, v=x2, x3=1-u-v) and use finite differences.
No third-party dependencies are required.
"""

from __future__ import annotations

import argparse
import cmath
import math
from dataclasses import dataclass


def _lin_class(value: float, *, tol: float) -> str:
    """3-way classification around 0 with tolerance."""
    v = float(value)
    t = float(tol)
    if v > t:
        return "pos"
    if v < -t:
        return "neg"
    return "near0"


def payoff_matrix_ab(a: float, b: float) -> tuple[tuple[float, float, float], ...]:
    """Return the 3x3 cyclic payoff matrix used by `DungeonAI(payoff_mode='matrix_ab')`."""
    aa = float(a)
    bb = float(b)
    return (
        (0.0, aa, -bb),
        (-bb, 0.0, aa),
        (aa, -bb, 0.0),
    )


def _dot3(x: tuple[float, float, float], y: tuple[float, float, float]) -> float:
    return float(x[0] * y[0] + x[1] * y[1] + x[2] * y[2])


def _matvec3(
    A: tuple[tuple[float, float, float], ...],
    x: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        _dot3(A[0], x),
        _dot3(A[1], x),
        _dot3(A[2], x),
    )


def _normalize_simplex3(x: tuple[float, float, float], *, eps: float = 1e-12) -> tuple[float, float, float]:
    x1, x2, x3 = (float(x[0]), float(x[1]), float(x[2]))
    x1 = max(float(eps), x1)
    x2 = max(float(eps), x2)
    x3 = max(float(eps), x3)
    s = x1 + x2 + x3
    if s <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (x1 / s, x2 / s, x3 / s)


def uv_to_simplex(u: float, v: float, *, eps: float = 1e-12) -> tuple[float, float, float]:
    """Convert (u,v) coords to a valid simplex point by clamping and renormalizing."""
    uu = float(u)
    vv = float(v)
    w3 = 1.0 - uu - vv
    return _normalize_simplex3((uu, vv, w3), eps=float(eps))


def simplex_to_uv(x: tuple[float, float, float]) -> tuple[float, float]:
    return (float(x[0]), float(x[1]))


def replicator_ode(x: tuple[float, float, float], *, a: float, b: float) -> tuple[float, float, float]:
    """Replicator ODE vector field for `matrix_ab` payoff."""
    A = payoff_matrix_ab(a, b)
    xx = _normalize_simplex3(x)
    Ax = _matvec3(A, xx)
    avg = _dot3(xx, Ax)
    return (
        xx[0] * (Ax[0] - avg),
        xx[1] * (Ax[1] - avg),
        xx[2] * (Ax[2] - avg),
    )


def replicator_ode_uv(u: float, v: float, *, a: float, b: float) -> tuple[float, float]:
    x = uv_to_simplex(u, v)
    dx = replicator_ode(x, a=float(a), b=float(b))
    return (float(dx[0]), float(dx[1]))


def lagged_discrete_update(
    *,
    p_cur: tuple[float, float, float],
    p_prev: tuple[float, float, float],
    a: float,
    b: float,
    selection_strength: float,
) -> tuple[float, float, float]:
    """Mean-field discrete update with one-round lag.

    p_{t+1} depends on (p_t, p_{t-1}). See module docstring.
    """
    k = float(selection_strength)
    if k < 0:
        raise ValueError("selection_strength must be >= 0")
    A = payoff_matrix_ab(a, b)
    pt = _normalize_simplex3(p_cur)
    ptm1 = _normalize_simplex3(p_prev)

    u = _matvec3(A, ptm1)
    u_bar = _dot3(pt, u)
    # Engine-faithful mean-field update:
    # - replicator_step produces positive weights per strategy
    # - the implied next distribution is proportional to current mass * weight
    w = (
        pt[0] * math.exp(k * (u[0] - u_bar)),
        pt[1] * math.exp(k * (u[1] - u_bar)),
        pt[2] * math.exp(k * (u[2] - u_bar)),
    )
    return _normalize_simplex3(w)


def lagged_discrete_update_uv(
    *,
    u: float,
    v: float,
    u_prev: float,
    v_prev: float,
    a: float,
    b: float,
    selection_strength: float,
) -> tuple[float, float]:
    p_cur = uv_to_simplex(u, v)
    p_prev = uv_to_simplex(u_prev, v_prev)
    p_next = lagged_discrete_update(
        p_cur=p_cur,
        p_prev=p_prev,
        a=float(a),
        b=float(b),
        selection_strength=float(selection_strength),
    )
    return simplex_to_uv(p_next)


def _fd_jacobian_2x2(f, x0: tuple[float, float], *, h: float = 1e-6) -> tuple[tuple[float, float], tuple[float, float]]:
    """Central-difference Jacobian for f: R^2 -> R^2."""
    h1 = float(h)
    h2 = float(h)
    x, y = (float(x0[0]), float(x0[1]))

    f_xp = f(x + h1, y)
    f_xm = f(x - h1, y)
    f_yp = f(x, y + h2)
    f_ym = f(x, y - h2)

    dfdx0 = (float(f_xp[0]) - float(f_xm[0])) / (2.0 * h1)
    dfdx1 = (float(f_xp[1]) - float(f_xm[1])) / (2.0 * h1)
    dfdy0 = (float(f_yp[0]) - float(f_ym[0])) / (2.0 * h2)
    dfdy1 = (float(f_yp[1]) - float(f_ym[1])) / (2.0 * h2)
    return ((dfdx0, dfdy0), (dfdx1, dfdy1))


def eigvals_2x2(J: tuple[tuple[float, float], tuple[float, float]]) -> tuple[complex, complex]:
    """Eigenvalues for a 2x2 matrix."""
    a, b = (complex(J[0][0]), complex(J[0][1]))
    c, d = (complex(J[1][0]), complex(J[1][1]))
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    s = cmath.sqrt(disc)
    return ((tr + s) / 2.0, (tr - s) / 2.0)


def _poly_eval(coeffs: list[complex], z: complex) -> complex:
    out = 0j
    for c in reversed(coeffs):
        out = out * z + c
    return out


def _durand_kerner_roots(coeffs: list[complex], *, max_iter: int = 200, tol: float = 1e-12) -> list[complex]:
    """Find all roots of a polynomial using Durand–Kerner.

    coeffs are increasing degree (c0 + c1 z + ... + cn z^n).
    """
    if len(coeffs) < 2:
        return []
    n = len(coeffs) - 1

    lead = complex(coeffs[-1])
    if abs(lead) == 0:
        raise ValueError("leading coefficient cannot be 0")
    cc = [complex(c) / lead for c in coeffs]

    R = 1.0 + max((abs(c) for c in cc[:-1]), default=0.0)
    roots = [R * cmath.exp(2j * math.pi * k / n) for k in range(n)]

    for _ in range(int(max_iter)):
        max_delta = 0.0
        new_roots: list[complex] = []
        for i in range(n):
            ri = roots[i]
            den = 1.0 + 0j
            for j in range(n):
                if j == i:
                    continue
                den *= (ri - roots[j])
            if abs(den) == 0:
                den = 1e-18 + 0j
            pi = _poly_eval(cc, ri)
            ri2 = ri - (pi / den)
            new_roots.append(ri2)
            max_delta = max(max_delta, abs(ri2 - ri))
        roots = new_roots
        if max_delta < float(tol):
            break
    return roots


def lagged_state_eigvals(
    *,
    A: tuple[tuple[float, float], tuple[float, float]],
    B: tuple[tuple[float, float], tuple[float, float]],
) -> list[complex]:
    """Eigenvalues of the 4D lagged state Jacobian from 2x2 blocks.

    Eigenvalues satisfy det(λ^2 I - λ A - B) = 0, a quartic polynomial.
    """
    a11, a12 = float(A[0][0]), float(A[0][1])
    a21, a22 = float(A[1][0]), float(A[1][1])
    b11, b12 = float(B[0][0]), float(B[0][1])
    b21, b22 = float(B[1][0]), float(B[1][1])

    # det([[λ^2 - a11 λ - b11, -(a12 λ + b12)], [-(a21 λ + b21), λ^2 - a22 λ - b22]])
    def poly_mul(p: list[complex], q: list[complex]) -> list[complex]:
        out = [0j] * (len(p) + len(q) - 1)
        for i, aa in enumerate(p):
            for j, bb in enumerate(q):
                out[i + j] += aa * bb
        return out

    def poly_sub(p: list[complex], q: list[complex]) -> list[complex]:
        out = [0j] * max(len(p), len(q))
        for i, c in enumerate(p):
            out[i] += c
        for i, c in enumerate(q):
            out[i] -= c
        return out

    m11 = [complex(-b11), complex(-a11), 1.0 + 0j]
    m22 = [complex(-b22), complex(-a22), 1.0 + 0j]
    m12 = [complex(-b12), complex(-a12)]
    m21 = [complex(-b21), complex(-a21)]

    poly = poly_sub(poly_mul(m11, m22), poly_mul(m12, m21))
    poly += [0j] * (5 - len(poly))
    poly = poly[:5]
    return _durand_kerner_roots(poly)


def rotation_sign_2d(delta: tuple[float, float], delta_next: tuple[float, float]) -> float:
    """Signed area (2D cross product z) for (delta -> delta_next)."""
    return float(delta[0] * delta_next[1] - delta[1] * delta_next[0])


@dataclass(frozen=True)
class ODEJacobianReport:
    J: tuple[tuple[float, float], tuple[float, float]]
    eigvals: tuple[complex, complex]
    rotation_z: float


@dataclass(frozen=True)
class ODESummary:
    alpha: float
    omega: float
    stability: str  # stable_focus | near_neutral_center | unstable_focus
    dominant: complex


@dataclass(frozen=True)
class LaggedJacobianReport:
    A: tuple[tuple[float, float], tuple[float, float]]
    B: tuple[tuple[float, float], tuple[float, float]]
    eigvals: list[complex]
    rotation_z: float


@dataclass(frozen=True)
class LaggedSummary:
    rho: float
    stability: str  # stable | near_neutral | unstable
    dominant: complex


def ode_jacobian_at_uniform(*, a: float, b: float, h: float = 1e-6) -> ODEJacobianReport:
    u0 = 1.0 / 3.0
    v0 = 1.0 / 3.0
    J = _fd_jacobian_2x2(lambda uu, vv: replicator_ode_uv(uu, vv, a=float(a), b=float(b)), (u0, v0), h=float(h))
    eigs = eigvals_2x2(J)

    delta = (1.0, 0.0)
    dt = 1e-3
    delta_next = (
        delta[0] + dt * (J[0][0] * delta[0] + J[0][1] * delta[1]),
        delta[1] + dt * (J[1][0] * delta[0] + J[1][1] * delta[1]),
    )
    rot = rotation_sign_2d(delta, delta_next)
    return ODEJacobianReport(J=J, eigvals=eigs, rotation_z=float(rot))


def summarize_ode(rep: ODEJacobianReport, *, tol: float = 1e-6) -> ODESummary:
    """Summarize ODE eigen-structure near the interior equilibrium.

    For the 2D reduced system we expect a conjugate pair. We report:
    - alpha: mean real part
    - omega: |imag| of the dominant eigenvalue
    - stability label based on alpha
    """
    lam1, lam2 = rep.eigvals
    # Prefer the eigenvalue with larger |imag| as the "dominant" rotation mode.
    dominant = lam1 if abs(lam1.imag) >= abs(lam2.imag) else lam2
    alpha = 0.5 * (float(lam1.real) + float(lam2.real))
    omega = float(abs(dominant.imag))
    cls = _lin_class(alpha, tol=float(tol))
    if cls == "neg":
        stability = "stable_focus"
    elif cls == "pos":
        stability = "unstable_focus"
    else:
        stability = "near_neutral_center"
    return ODESummary(alpha=float(alpha), omega=float(omega), stability=stability, dominant=dominant)


def lagged_jacobian_at_uniform(*, a: float, b: float, selection_strength: float, h: float = 1e-6) -> LaggedJacobianReport:
    u0 = 1.0 / 3.0
    v0 = 1.0 / 3.0

    def g_cur(uu: float, vv: float) -> tuple[float, float]:
        return lagged_discrete_update_uv(
            u=uu,
            v=vv,
            u_prev=u0,
            v_prev=v0,
            a=float(a),
            b=float(b),
            selection_strength=float(selection_strength),
        )

    def g_prev(uu_prev: float, vv_prev: float) -> tuple[float, float]:
        return lagged_discrete_update_uv(
            u=u0,
            v=v0,
            u_prev=uu_prev,
            v_prev=vv_prev,
            a=float(a),
            b=float(b),
            selection_strength=float(selection_strength),
        )

    Ablk = _fd_jacobian_2x2(lambda uu, vv: g_cur(uu, vv), (u0, v0), h=float(h))
    Bblk = _fd_jacobian_2x2(lambda uu, vv: g_prev(uu, vv), (u0, v0), h=float(h))
    eigs = lagged_state_eigvals(A=Ablk, B=Bblk)

    # Under the one-round lag, p_{t+1} is primarily driven by p_{t-1}.
    # Measuring rotation using a perturbation in p_{t-1} avoids the trivial
    # zero response at the uniform equilibrium (where d/d p_t can vanish).
    delta_prev = (1e-4, 0.0)
    base = lagged_discrete_update_uv(
        u=u0,
        v=v0,
        u_prev=u0,
        v_prev=v0,
        a=float(a),
        b=float(b),
        selection_strength=float(selection_strength),
    )
    pert = lagged_discrete_update_uv(
        u=u0,
        v=v0,
        u_prev=u0 + delta_prev[0],
        v_prev=v0 + delta_prev[1],
        a=float(a),
        b=float(b),
        selection_strength=float(selection_strength),
    )
    delta_next = (pert[0] - base[0], pert[1] - base[1])
    rot = rotation_sign_2d(delta_prev, delta_next)

    return LaggedJacobianReport(A=Ablk, B=Bblk, eigvals=eigs, rotation_z=float(rot))


def summarize_lagged(rep: LaggedJacobianReport, *, tol: float = 1e-6) -> LaggedSummary:
    """Summarize lagged (2-step) linearization around uniform equilibrium.

    We report rho = max |lambda| for the 4D state transition and a stability label
    around the discrete-time boundary rho=1.
    """
    if not rep.eigvals:
        return LaggedSummary(rho=0.0, stability="stable", dominant=0j)
    dominant = max(rep.eigvals, key=lambda z: abs(z))
    rho = float(abs(dominant))
    cls = _lin_class(rho - 1.0, tol=float(tol))
    if cls == "neg":
        stability = "stable"
    elif cls == "pos":
        stability = "unstable"
    else:
        stability = "near_neutral"
    return LaggedSummary(rho=float(rho), stability=stability, dominant=dominant)


def _fmt_c(z: complex) -> str:
    return f"{z.real:+.6g}{z.imag:+.6g}j"


def main() -> None:
    p = argparse.ArgumentParser(description="Jacobian + local rotation checks for matrix_ab mean-field dynamics")
    p.add_argument("--a", type=float, required=True)
    p.add_argument("--b", type=float, required=True)
    p.add_argument("--selection-strength", type=float, default=0.02)
    p.add_argument("--mode", type=str, default="lagged", choices=["lagged", "ode"])
    p.add_argument("--fd-h", type=float, default=1e-6)
    p.add_argument("--summary", action="store_true", help="Print a compact stability summary (alpha/omega or rho).")
    p.add_argument(
        "--summary-tol",
        type=float,
        default=1e-6,
        help="Tolerance used for near-neutral classification (alpha~0 or rho~1).",
    )
    args = p.parse_args()

    aa = float(args.a)
    bb = float(args.b)
    h = float(args.fd_h)

    print("=== Jacobian rotation check (mean-field) ===")
    print(f"matrix_ab a={aa} b={bb}")
    print("equilibrium (uniform): p*=(1/3,1/3,1/3)")

    if str(args.mode) == "ode":
        rep = ode_jacobian_at_uniform(a=aa, b=bb, h=h)
        print("\n[ODE] J (uv coords) at p*:")
        print(f"  [{rep.J[0][0]:+.6g}  {rep.J[0][1]:+.6g}]")
        print(f"  [{rep.J[1][0]:+.6g}  {rep.J[1][1]:+.6g}]")
        print("  eigvals:")
        for ev in rep.eigvals:
            print(f"    {_fmt_c(ev)}")
        print(f"  rotation_z (Euler dt=1e-3, delta=(1,0)): {rep.rotation_z:+.6g}")
        if bool(args.summary):
            s = summarize_ode(rep, tol=float(args.summary_tol))
            print("  summary:")
            print(f"    alpha={s.alpha:+.6g}  omega={s.omega:.6g}  stability={s.stability}")
            print(f"    dominant={_fmt_c(s.dominant)}")
        return

    k = float(args.selection_strength)
    rep2 = lagged_jacobian_at_uniform(a=aa, b=bb, selection_strength=k, h=h)
    print(f"\n[Lagged discrete] selection_strength={k:g}")
    print("  Blocks for p_{t+1} = A p_t + B p_{t-1} (in uv coords):")
    print("  A=")
    print(f"    [{rep2.A[0][0]:+.6g}  {rep2.A[0][1]:+.6g}]")
    print(f"    [{rep2.A[1][0]:+.6g}  {rep2.A[1][1]:+.6g}]")
    print("  B=")
    print(f"    [{rep2.B[0][0]:+.6g}  {rep2.B[0][1]:+.6g}]")
    print(f"    [{rep2.B[1][0]:+.6g}  {rep2.B[1][1]:+.6g}]")
    print("  state eigvals (quartic):")
    for ev in sorted(rep2.eigvals, key=lambda z: abs(z), reverse=True):
        print(f"    {_fmt_c(ev)}  |λ|={abs(ev):.6g}")
    print(f"  rotation_z (perturb p[t-1], one-step response): {rep2.rotation_z:+.6g}")
    if bool(args.summary):
        s2 = summarize_lagged(rep2, tol=float(args.summary_tol))
        print("  summary:")
        print(f"    rho={s2.rho:.6g}  stability={s2.stability}")
        print(f"    dominant={_fmt_c(s2.dominant)}")


if __name__ == "__main__":
    main()
